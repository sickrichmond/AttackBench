import argparse
import wandb
from pathlib import Path

# Configuration
ENTITY = "attackbench"
PROJECT = "attackbench-precompiled-distancies"

def upload_all_compiled(results_dir):
    results_path = Path(results_dir)
    
    # Iterate over all JSON files in the directory
    for file_path in results_path.glob("*.json"):
        # Heuristic check to skip unrelated JSONs
        if file_path.name.count("-") < 3:
            continue

        print(f"Processing: {file_path.name}")
        
        # Parse metadata from filename: dataset-threat-model-batch.json
        try:
            parts = file_path.stem.split("-")
            dataset = parts[0]
            threat_model = parts[1]
            batch_size = parts[-1]
            # Reconstruct model name (handles cases with hyphens)
            model_name = "-".join(parts[2:-1]) 
            
            artifact_name = file_path.stem
            
            # Initialize run for upload
            with wandb.init(project=PROJECT, entity=ENTITY, job_type="upload-reference") as run:
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="reference_distances",
                    metadata={
                        "dataset": dataset,
                        "model": model_name,
                        "threat_model": threat_model,
                        "batch_size": batch_size
                    }
                )
                artifact.add_file(str(file_path))
                run.log_artifact(artifact)
                
            print(f"Successfully uploaded: {artifact_name}")
            
        except Exception as e:
            print(f"Failed to upload {file_path.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload compiled distances to W&B")
    parser.add_argument("--dir", default="results", help="Directory containing compiled .json files")
    args = parser.parse_args()
    
    upload_all_compiled(args.dir)