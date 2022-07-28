from pathlib import Path
from datetime import datetime


home_path = Path(__file__).resolve().parent.parent
data_folder = Path(home_path, "data/")

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
results_folder = Path(home_path, f"results", f"experiment_{date}")