"""
MIA Mitigation Strategies in Federated Learning:
A PyTorch app that utilizes Flower.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from opacus import PrivacyEngine
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    RandomErasing,
)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, dp_on=False):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.GroupNorm(num_groups=1, num_channels=in_planes) if dp_on else nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=not dp_on)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.GroupNorm(num_groups=1, num_channels=out_planes) if dp_on else nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=not dp_on)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.drop_rate = drop_rate
        self.equal_in_out = in_planes == out_planes
        self.conv_shortcut = (
            (not self.equal_in_out)
            and nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False
            )
            or None
        )

    def forward(self, x):
        if not self.equal_in_out:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_in_out else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equal_in_out else self.conv_shortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, dp_on=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, dp_on
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, dp_on):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    drop_rate,
                    dp_on,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0, dp_on=False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], BasicBlock, 1, drop_rate, dp_on
        )
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], BasicBlock, 2, drop_rate, dp_on
        )
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], BasicBlock, 2, drop_rate, dp_on
        )
        self.bn1 = nn.GroupNorm(num_groups=1, num_channels=nChannels[3]) if dp_on else nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=not dp_on)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif dp_on and isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif not dp_on and isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


fds = None

def load_data(
    partition_id: int,
    num_partitions: int,
    test_size: float = 0.2,
    seed: int = 42,
    batch_size: int = 128,
    num_workers: int = 4,
    random_crop_padding: int = 4,
    random_erasing_probability: float = 0.5,
    cifar100_mean: tuple = (0.5071, 0.4867, 0.4409),
    cifar100_std: tuple = (0.2675, 0.2565, 0.2761),
):

    def apply_transforms_train(batch):
        """Apply transforms to the partition from FederatedDataset for training."""
        train_transforms = Compose(
            [
                RandomCrop(32, padding=random_crop_padding),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(cifar100_mean, cifar100_std),
                RandomErasing(p=random_erasing_probability),
            ]
        )
        batch["img"] = [train_transforms(img) for img in batch["img"]]
        if "fine_label" in batch:
            batch["label"] = batch["fine_label"]

        return batch

    def apply_transforms_test(batch):
        """Apply transforms to the partition from FederatedDataset for testign."""
        test_transforms = Compose(
            [
                ToTensor(),
                Normalize(cifar100_mean, cifar100_std),
            ]
        )
        batch["img"] = [test_transforms(img) for img in batch["img"]]

        if "fine_label" in batch:
            batch["label"] = batch["fine_label"]

        return batch

    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar100",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=test_size, seed=seed)

    train_split = partition_train_test["train"].with_transform(apply_transforms_train)
    test_split = partition_train_test["test"].with_transform(apply_transforms_test)
    cuda = torch.cuda.is_available()
    trainloader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if cuda else 0,
        pin_memory=cuda,
        persistent_workers=cuda,
    )
    testloader = DataLoader(
        test_split,
        batch_size=batch_size,
        num_workers=num_workers if cuda else 0,
        pin_memory=cuda,
        persistent_workers=cuda,
    )
    return trainloader, testloader


def train(
    net,
    trainloader,
    epochs,
    lr,
    device,
    label_smoothing: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    dp_on: bool = False,
    noise_multiplier: float = 0.5,
):
    """Train the model on the training set."""
    print(
        f"Using GPU: {torch.cuda.get_device_name(device)}"
        if torch.cuda.is_available()
        else "Using CPU"
    )
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    if dp_on:
        privacy_engine = PrivacyEngine()
        net, optimizer, trainloader = privacy_engine.make_private(
            module=net,
            optimizer=optimizer,
            data_loader=trainloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            if not dp_on:
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
