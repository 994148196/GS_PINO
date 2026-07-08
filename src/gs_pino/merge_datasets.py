"""Merge multiple freegs datasets into a single larger dataset."""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Input dataset files to merge")
    parser.add_argument("--output", required=True, help="Output merged dataset file")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Merging Datasets")
    print(f"{'='*60}")
    
    all_arrays = {}
    keys_to_merge = ["R", "Z", "psi_total", "psi_plasma", "psi_coils", "mask", 
                     "rhs", "coil_currents", "params", "axes", "L", "Beta0"]
    
    for i, input_path in enumerate(args.inputs):
        print(f"  Loading: {input_path}")
        data = np.load(input_path)
        
        if i == 0:
            for key in keys_to_merge:
                if key in data:
                    all_arrays[key] = [data[key]]
                else:
                    all_arrays[key] = []
        else:
            for key in keys_to_merge:
                if key in data:
                    all_arrays[key].append(data[key])

    merged = {}
    total_samples = 0
    for key, arrays in all_arrays.items():
        if len(arrays) > 0:
            if key == "R" or key == "Z":
                merged[key] = np.concatenate(arrays, axis=0)
            else:
                merged[key] = np.concatenate(arrays, axis=0)
            total_samples = merged[key].shape[0]
            print(f"  {key}: {merged[key].shape}")

    print(f"\n  Total samples: {total_samples}")
    
    np.savez_compressed(args.output, **merged)
    print(f"  Saved to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()