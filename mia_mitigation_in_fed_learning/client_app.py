"""MIA-Mitigation-in-Fed-Learning: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from mia_mitigation_in_fed_learning.task import WideResNet, load_data
from mia_mitigation_in_fed_learning.task import test as test_fn
from mia_mitigation_in_fed_learning.task import train as train_fn

# Flower ClientApp
app = ClientApp()


def create_model(context: Context):
    return WideResNet(
        depth=context.run_config["model-depth"],
        widen_factor=context.run_config["model-widen-factor"],
        num_classes=context.run_config["model-num-classes"],
        drop_rate=context.run_config["drop-rate"],
        dp_on=context.run_config["dp-on"],
    )


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data_client(context: Context):
    return load_data(
        partition_id=context.node_config["partition-id"],
        num_partitions=context.node_config["num-partitions"],
        train_size=context.run_config["train-split-size"],
        val_size=context.run_config["val-split-size"],
        test_size=context.run_config["test-split-size"],
        alpha=context.run_config["alpha"],
        seed=context.run_config["data-partition-seed"],
        batch_size=context.run_config["batch-size"],
        num_workers=context.run_config["num-workers"],
        random_crop_padding=context.run_config["random-crop-padding"],
        random_erasing_probability=context.run_config["random-erasing-probability"],
        cifar100_mean=tuple(
            float(x) for x in context.run_config["cifar-100-mean"].split(",")
        ),
        cifar100_std=tuple(
            float(x) for x in context.run_config["cifar-100-std"].split(",")
        ),
    )


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = create_model(context)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    # Load the data
    trainloader, _, _ = load_data_client(context)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        label_smoothing=context.run_config["label-smoothing"],
        momentum=context.run_config["momentum"],
        weight_decay=context.run_config["weight-decay"],
        max_grad_norm=context.run_config["max-grad-norm"],
        dp_on=context.run_config["dp-on"],
        noise_multiplier=(
            context.run_config["noise-multiplier"]
            if context.run_config["dp-on"]
            else None
        ),
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = create_model(context)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    # Load the data
    _, valloader, _ = load_data_client(context)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
