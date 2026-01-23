#!/usr/bin/env python3
"""
Plot all normalized PDD files from validation/ folder.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_pdd_file(filepath):
    """Read PDD data from file, skipping header lines."""
    depths = []
    doses_norm = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                depths.append(float(parts[0]))
                doses_norm.append(float(parts[2]))

    return np.array(depths), np.array(doses_norm)

def main():
    validation_dir = '/workspaces/SM_2D/validation'

    # Find all PDD files
    pdd_files = sorted([f for f in os.listdir(validation_dir) if f.startswith('pdd_') and f.endswith('.txt')])

    if not pdd_files:
        print("No PDD files found in validation/ folder")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each PDD
    for pdd_file in pdd_files:
        filepath = os.path.join(validation_dir, pdd_file)
        depths, doses_norm = read_pdd_file(filepath)

        # Extract energy from filename
        energy = pdd_file.split('_')[1].replace('MeV', '')
        label = f'{energy} MeV'

        ax.plot(depths, doses_norm, label=label, linewidth=2)

    ax.set_xlabel('Depth (mm)', fontsize=12)
    ax.set_ylabel('Normalized Dose', fontsize=12)
    ax.set_title('Normalized PDD Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('/workspaces/SM_2D/validation/pdd_comparison.png', dpi=150)
    print("Plot saved to validation/pdd_comparison.png")

if __name__ == '__main__':
    main()
