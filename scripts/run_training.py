"""
DepthWizard Training Runner
CLI entry point to launch Depth Anything V2 fine-tuning on ISPRS Potsdam and GAMUS datasets.
"""

import sys
import os

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.train_depth_anything import parse_args, train

if __name__ == "__main__":
    args = parse_args()
    train(args)
