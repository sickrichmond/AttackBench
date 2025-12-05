"""
Legacy upload script - now uses the new wandb_manager.
Maintained for backward compatibility.
"""
import argparse
from pathlib import Path
import sys

# Add project root to path to import wandb_manager
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from attackbench.wandb_manager import upload_directory

def upload_all_compiled(results_dir):
    """
    Legacy function - now delegates to new wandb_manager.
    """
    print(f"Using new W&B manager to upload from: {results_dir}")
    
    results = upload_directory(
        directory=results_dir,
        dataset=None,  # Upload all datasets
        overwrite=False
    )
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"Legacy upload completed: {successful}/{total} files uploaded")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload compiled distances to W&B")
    parser.add_argument("--dir", default="results", help="Directory containing compiled .json files")
    args = parser.parse_args()
    
    upload_all_compiled(args.dir)