#!/usr/bin/env python3
"""
Renumber synthetic data sample directories by adding an offset.

Usage:
    python scripts/utils/renumber_data_files.py --data data/synthetic_optimized_2 --offset 200
"""

import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Renumber synthetic data directories')
    parser.add_argument('--data', type=str, required=True,
                        help='Directory containing sample_* folders')
    parser.add_argument('--offset', type=int, default=200,
                        help='Number to add to each folder index')
    args = parser.parse_args()

    base_dir = args.data
    offset = args.offset

    if not os.path.isdir(base_dir):
        print(f"Error: directory not found: {base_dir}")
        return

    for name in sorted(os.listdir(base_dir)):
        old_path = os.path.join(base_dir, name)

        if not os.path.isdir(old_path):
            continue

        try:
            prefix, number_str = name.rsplit("_", 1)
            number = int(number_str)
        except ValueError:
            continue

        new_number = number + offset
        new_name = f"{prefix}_{new_number:0{len(number_str)}d}"
        new_path = os.path.join(base_dir, new_name)

        print(f"Renaming: {name} -> {new_name}")
        os.rename(old_path, new_path)


if __name__ == '__main__':
    main()
