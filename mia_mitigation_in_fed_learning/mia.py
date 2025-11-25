import argparse
import numpy as np
import torch
import torch.nn as nn

from mia_mitigation_in_fed_learning.task import WideResNet, load_data
from mia_mitigation_in_fed_learning.client_app import get_device

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from art.estimators.classification import PyTorchClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox


def load_model(
    model_depth: int,
    num_classes: int,
    widen_factor: int,
    drop_rate: float,
    dp_on: bool,
    model_path: str,
):
    model = WideResNet(model_depth, num_classes, widen_factor, drop_rate, dp_on)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def prepare_data(num_partitions: int):
    # Load in data, which is split 50:20:30 already and mark 50% as members and 50% as non-members
    member_images, member_labels = [], []
    non_member_images, non_member_labels = [], []

    for i in range(num_partitions):
        trainloader, valloader, testloader = load_data(
            partition_id=i, num_partitions=num_partitions
        )
        for batch in trainloader:
            member_images.extend(batch["img"].cpu().numpy())
            member_labels.extend(batch["label"].cpu().numpy())
        for batch in valloader:
            non_member_images.extend(batch["img"].cpu().numpy())
            non_member_labels.extend(batch["label"].cpu().numpy())
        for batch in testloader:
            non_member_images.extend(batch["img"].cpu().numpy())
            non_member_labels.extend(batch["label"].cpu().numpy())

    return (
        np.array(member_images),
        np.array(member_labels),
        np.array(non_member_images),
        np.array(non_member_labels),
    )


def run_mia(
    classifier: PyTorchClassifier,
    member_x: np.ndarray,
    member_y: np.ndarray,
    nonmember_x: np.ndarray,
    nonmember_y: np.ndarray,
):

    attack_train_size = len(member_x) // 2
    attack_member_x = member_x[:attack_train_size]
    attack_member_y = member_y[:attack_train_size]
    attack_nonmember_x = nonmember_x[:attack_train_size]
    attack_nonmember_y = nonmember_y[:attack_train_size]

    eval_member_x = member_x[attack_train_size:]
    eval_member_y = member_y[attack_train_size:]
    eval_nonmember_x = nonmember_x[attack_train_size:]
    eval_nonmember_y = nonmember_y[attack_train_size:]

    eval_x = np.concatenate([eval_member_x, eval_nonmember_x], axis=0)
    eval_y = np.concatenate([eval_member_y, eval_nonmember_y], axis=0)
    true_membership = np.concatenate(
        [np.ones(len(eval_member_x)), np.zeros(len(eval_nonmember_x))]
    )

    attack = MembershipInferenceBlackBox(
        estimator=classifier,
        attack_model_type="nn",
    )
    attack.fit(
        x=attack_member_x,
        y=attack_member_y,
        test_x=attack_nonmember_x,
        test_y=attack_nonmember_y,
    )

    inferred_membership = attack.infer(eval_x, eval_y)
    predicted_membership = (inferred_membership >= 0.5).astype(int)
    accuracy = accuracy_score(true_membership, predicted_membership)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_membership, predicted_membership, average="binary"
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-depth", type=int, default=28)
    parser.add_argument("--widen-factor", type=int, default=4)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--dp-on", type=bool, default=False)
    parser.add_argument("--model-path", type=str, default="final_model.pt")
    parser.add_argument("--num-partitions", type=int, default=10)
    args = parser.parse_args()

    print("Loading Target Model")
    model = load_model(
        model_depth=args.model_depth,
        num_classes=args.num_classes,
        widen_factor=args.widen_factor,
        drop_rate=args.drop_rate,
        dp_on=args.dp_on,
        model_path=args.model_path,
    )

    print("Preparing Data for MIA")
    member_x, member_y, non_member_x, non_member_y = prepare_data(
        num_partitions=args.num_partitions
    )

    print("Wrapping model for ART MIA")
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=args.num_classes,
        device_type="gpu" if get_device() == "cuda" else "cpu",
    )

    print("Running ART MembershipInferenceBlackBox Attack")
    res_art = run_mia(
        classifier=classifier,
        member_x=member_x,
        member_y=member_y,
        nonmember_x=non_member_x,
        nonmember_y=non_member_y,
    )

    print("ART MembershipInferenceBlackBox Results:")
    print(f"Accuracy: {res_art['accuracy']}")
    print(f"Precision: {res_art['precision']}")
    print(f"Recall: {res_art['recall']}")
    print(f"F1 score: {res_art['f1_score']}")
