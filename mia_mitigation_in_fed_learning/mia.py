import argparse
import torch

from mia_mitigation_in_fed_learning.task import WideResNet


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


def prepare_data():
    pass


def run_mia():
    pass


def save_results():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-depth", type=int, default=28)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--widen-factor", type=int, default=4)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--dp-on", type=bool, default=False)
    parser.add_argument("--model-path", type=str, default="final_model.pt")
    args = parser.parse_args()

    model = load_model(
        model_depth=args.model_depth,
        num_classes=args.num_classes,
        widen_factor=args.widen_factor,
        drop_rate=args.drop_rate,
        dp_on=args.dp_on,
        model_path=args.model_path,
    )
    prepare_data()
    run_mia()
    save_results()


if __name__ == "__main__":
    main()
