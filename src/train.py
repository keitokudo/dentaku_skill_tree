import argparse
from dentaku_trainer import DentakuTrainer


def main(args):
    trainer = DentakuTrainer(args, mode=args.mode)
    trainer(train_only=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = DentakuTrainer.add_args(parser)
    parser.add_argument(
        "--mode",
        help="Select mode",
        type=str,
        required=True,
        choices=["train", "resume"]
    )
    args = parser.parse_args()
    main(args)
