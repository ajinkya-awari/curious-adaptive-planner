"""
run_all.py  —  Runs the full pipeline: experiment → figures.

Usage: python run_all.py
"""

import subprocess
import sys
import time


def run(script: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  ▶  {script}")
    print(f"{'─' * 60}")
    t0 = time.time()
    subprocess.run([sys.executable, script], check=True)
    print(f"  ✓  finished in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    run('experiment.py')
    run('visualize.py')
    print('\n' + '─' * 60)
    print('  Pipeline complete.')
    print('  Results : results/')
    print('  Figures : results/figures/')
    print('─' * 60)
