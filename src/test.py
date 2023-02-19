import argparse
from pathlib import Path

from dentaku_trainer import DentakuTrainer


def main(args):
    trainer = DentakuTrainer(args, mode="test")
    trainer.test()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = DentakuTrainer.add_args(parser)
    args = parser.parse_args()
    main(args)
