from pathlib import Path
from datetime import datetime


home_path = Path(__file__).resolve().parent.parent
data_folder = Path(home_path, "data/")

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
main_result_folder = "results"
results_folder = Path(home_path, main_result_folder, f"experiment_{date}")