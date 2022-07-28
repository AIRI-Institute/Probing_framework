from pathlib import Path
from datetime import datetime


home_path = Path(__file__).resolve().parent.parent
data_folder = Path(home_path, "data/")

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
res_name = f"results_{date}/"
results_folder = Path(home_path, res_name)