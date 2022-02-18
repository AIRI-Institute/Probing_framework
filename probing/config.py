from pathlib import Path


home_path = Path(__file__).resolve().parent.parent
data_folder = Path(home_path, "data/")
results_folder = Path(home_path, "results/")