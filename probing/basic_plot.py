import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

try:
    from typing import Literal  # type: ignore
except:
    from typing_extensions import Literal  # type: ignore

from transformers.utils import logging

logger = logging.get_logger("probing")


class BasicPlot:
    PARAMS_FIELD = "params"
    RESULTS_FIELD = "results"

    LANGUAGE_FIELD = "task_language"
    TASK_FIELD = "task_category"
    MODEL_NAME_FIELD = "hf_model_name"
    CLASSIFIER_FIELD = "classifier_name"
    METRIC_FIELD = "metric_names"

    def __init__(
        self,
        x_field: str = "layer",
        y_field: str = "task_category",
        value_field: str = "metric_scores",
    ):
        self.x_field = x_field
        self.y_field = y_field
        self.value_field = value_field

    @staticmethod
    def get_logs(paths: List[Path]) -> List[Path]:
        logs_path = []
        for path in paths:
            internal_log_paths = path.glob("**/*/log.json")
            for log_path in internal_log_paths:
                if log_path not in logs_path:
                    logs_path.append(log_path)
        return logs_path

    @lru_cache()
    def aggregation(
        self,
        res_paths: Union[Path, List[Path]],
        metric_name: Literal["f1", "accuracy"] = "f1",
        stage: Literal["val", "test"] = "test",
    ) -> pd.DataFrame:
        aggregated_data_dict: Dict[Any, Any] = {
            BasicPlot.LANGUAGE_FIELD: [],
            BasicPlot.TASK_FIELD: [],
            BasicPlot.MODEL_NAME_FIELD: [],
            BasicPlot.CLASSIFIER_FIELD: [],
            BasicPlot.METRIC_FIELD: [],
            "layer": [],
            "metric_scores": [],
            "log_path": [],
        }

        if not isinstance(res_paths, list):
            res_paths = [res_paths]
        res_paths = [Path(path).resolve() for path in res_paths]

        log_paths = BasicPlot.get_logs(res_paths)
        if len(log_paths) == 0:
            logger.warning("None logs were found for the given paths.")

        for log_path in log_paths:
            with open(log_path) as f:
                data = json.load(f)

            params = data[BasicPlot.PARAMS_FIELD]
            all_results = data[BasicPlot.RESULTS_FIELD]

            lang = params[BasicPlot.LANGUAGE_FIELD]
            task_category = params[BasicPlot.TASK_FIELD]
            model_name = params[BasicPlot.MODEL_NAME_FIELD]
            classifier_name = params[BasicPlot.CLASSIFIER_FIELD]
            stage_scores = all_results[f"{stage}_score"][metric_name]

            for layer_num, stage_res in stage_scores.items():
                layer = int(layer_num) + 1

                if isinstance(stage_res, list):
                    aggregated_scores = np.mean(stage_res)
                else:
                    raise NotImplementedError()

                aggregated_data_dict[BasicPlot.LANGUAGE_FIELD].append(lang)
                aggregated_data_dict[BasicPlot.TASK_FIELD].append(task_category)
                aggregated_data_dict[BasicPlot.MODEL_NAME_FIELD].append(model_name)
                aggregated_data_dict[BasicPlot.CLASSIFIER_FIELD].append(classifier_name)
                aggregated_data_dict[BasicPlot.METRIC_FIELD].append(metric_name)

                aggregated_data_dict["layer"].append(layer)
                aggregated_data_dict["metric_scores"].append(aggregated_scores)
                aggregated_data_dict["log_path"].append(str(log_path))

        return pd.DataFrame(aggregated_data_dict)

    def make_pivot_table(self, aggregated_data_df: pd.DataFrame) -> pd.DataFrame:
        pivot_table = pd.pivot_table(
            aggregated_data_df,
            columns=[self.x_field],
            index=[self.y_field],
            values=[self.value_field],
        )
        pivot_table.columns = pivot_table.columns.droplevel()
        return pivot_table

    def plot(
        self,
        aggregated_data_df: pd.DataFrame,
        metric_name: Literal["f1", "accuracy"] = "f1",
        stage: Literal["val", "test"] = "test",
    ) -> None:
        pivot_table = self.make_pivot_table(aggregated_data_df)

        fig, ax = plt.subplots(1, 1, figsize=(40, 30))

        sns.heatmap(
            pivot_table,
            ax=ax,
            square=True,
            cbar=True,
            vmax=1,
            vmin=0,
            linewidths=1.01,
            cbar_kws={"shrink": 0.5, "label": f"Metric - {metric_name}", "pad": 0.005},
        )

        model_name = aggregated_data_df[BasicPlot.MODEL_NAME_FIELD][0]

        ax.set_ylabel(self.y_field.upper(), fontsize=40)
        ax.set_xlabel(self.x_field.upper(), fontsize=40)
        ax.set_title(f'Layerwise probing for "{model_name}"', fontsize=40)
        ax.tick_params(axis="both", labelsize=30)
        cbar = ax.collections[0].colorbar

        cbar.ax.tick_params(labelsize=30)
        cbar.ax.tick_params(labelsize=30, axis="y", rotation=0)
        ax.figure.axes[-1].yaxis.label.set_size(40)
        # fig.savefig("heatmap_best.pdf", bbox_inches="tight", dpi=500)
