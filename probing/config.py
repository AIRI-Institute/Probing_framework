from datetime import datetime
from pathlib import Path

home_path = Path(__file__).resolve().parent.parent
data_folder = Path(home_path, "data/")

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
MAIN_RES_FOLDER = "results"
results_folder = Path(home_path, MAIN_RES_FOLDER, f"experiment_{date}")
