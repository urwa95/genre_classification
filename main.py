import os
import mlflow
import hydra
from omegaconf import DictConfig

# Absolute path to this script's directory (repo root)
top_dir = os.path.abspath(os.path.dirname(__file__))

@hydra.main(config_path="conf", config_name="config")
def go(cfg: DictConfig):
    runs = {}
    project = cfg.main.project_name

    # Step 1: Download raw data
    if "download" in cfg.main.execute_steps:
        runs["download"] = mlflow.run(
            uri=os.path.join(top_dir, "download"),
            entry_point="main",
            parameters={
                "file_url":            cfg.data.file_url,
                "artifact_name":       cfg.download.output_artifact,
                "artifact_description":"Raw dataset downloaded from source",
            },
        )

    # Step 2: Preprocess downloaded data
    if "preprocess" in cfg.main.execute_steps:
        runs["preprocess"] = mlflow.run(
            uri=os.path.join(top_dir, "preprocess"),
            entry_point="main",
            parameters={
                "input_artifact":       cfg.preprocess.input_artifact,
                "artifact_name":        cfg.preprocess.output_artifact.split(":")[0],
                "artifact_type":        cfg.preprocess.artifact_type,
                "artifact_description": "Data after preprocessing (cleaning, imputing, feature derivation)",
            },
        )

    # Step 3: Check data quality
    if "check_data" in cfg.main.execute_steps:
        runs["check_data"] = mlflow.run(
            uri=os.path.join(top_dir, "check_data"),
            entry_point="main",
            parameters={
                "project_name":    project,
                "input_artifact":  cfg.check_data.input_artifact,
                "output_artifact": cfg.check_data.output_artifact,
                "artifact_type":   cfg.check_data.artifact_type,
            },
        )

    # Step 4: Split into train/test
    if "segregate" in cfg.main.execute_steps:
        runs["segregate"] = mlflow.run(
            uri=os.path.join(top_dir, "segregate"),
            entry_point="main",
            parameters={
                "project_name":    project,
                "input_artifact":  cfg.segregate.input_artifact,
                "output_artifact": cfg.segregate.output_artifact,
                "artifact_type":   cfg.segregate.artifact_type,
            },
        )

    # Step 5: Train and export model
    if "random_forest" in cfg.main.execute_steps:
        runs["random_forest"] = mlflow.run(
            uri=os.path.join(top_dir, "random_forest"),
            entry_point="main",
            parameters={
                "project_name":    project,
                "input_artifact":  cfg.random_forest.input_artifact,
                "output_artifact": cfg.random_forest.output_artifact,
                "artifact_type":   cfg.random_forest.artifact_type,
            },
        )

    # Step 6: Evaluate exported model
    if "evaluate" in cfg.main.execute_steps:
        runs["evaluate"] = mlflow.run(
            uri=os.path.join(top_dir, "evaluate"),
            entry_point="main",
            parameters={
                "project_name": project,
                "model_export": cfg.evaluate.model_export,
                "test_data":    cfg.evaluate.test_data,
            },
        )

if __name__ == "__main__":
    go()
