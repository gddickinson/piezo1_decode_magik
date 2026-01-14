#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 15:27:33 2026

@author: george
"""

import os

base_dir = "/Users/george/Documents/python_projects/piezo1_decode_magik/data/synthetic_optimized_2"
offset = 200

for name in os.listdir(base_dir):
    old_path = os.path.join(base_dir, name)

    # Only process directories
    if not os.path.isdir(old_path):
        continue

    # Split prefix and number
    try:
        prefix, number_str = name.rsplit("_", 1)
        number = int(number_str)
    except ValueError:
        # Skip folders that don't match the pattern
        continue

    # Add offset and preserve zero padding
    new_number = number + offset
    new_name = f"{prefix}_{new_number:0{len(number_str)}d}"

    new_path = os.path.join(base_dir, new_name)

    print(f"Renaming: {name} → {new_name}")
    os.rename(old_path, new_path)

