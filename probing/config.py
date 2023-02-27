from datetime import datetime
from pathlib import Path

HOME_PATH = Path(__file__).resolve().parent.parent
DATA_FOLDER_PATH = Path(HOME_PATH, "data/")

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
MAIN_RES_FOLDER = "results"
results_folder = Path(HOME_PATH, MAIN_RES_FOLDER, f"experiment_{date}")
