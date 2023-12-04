from probing.pipeline import ProbingPipeline
from pathlib import Path
import os
import torch


def main():
    experiment = ProbingPipeline(
            hf_model_name="bigcode/santacoder",
            device="cuda:0",
            metric_names=["f1", "accuracy", "classification_report"],
            encoding_batch_size=4,
            classifier_batch_size=16)
    
    tasks = ["COGNITIVE_COMPLEXITY"]
    language = "java"

    for probing_task in tasks:

        print(f"Calculating for {probing_task}..")

        filepath = os.path.join("../../../probing_data", language + "_" + probing_task + ".csv")

        experiment.run(probe_task=Path(filepath).stem,
                       path_to_task_file=filepath,
                       verbose=True,
                       train_epochs=10,)

        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    main()