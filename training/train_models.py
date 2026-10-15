import subprocess
import sys
import os

def run_script(script_path):
    """Run a Python script."""
    result = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(script_path))
    return result.returncode

if __name__ == "__main__":
    # Run data preprocessing
    print("Preprocessing data...")
    run_script('../utils/data_preprocessing.py')

    # Run experiments and save final models
    print("Running experiment pipeline...")
    run_script('experiments.py')

    print("Experiment pipeline complete!")