import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser

from src.utils.WindowEventsParser import WindowEventsParser
from datetime import datetime, timedelta


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--result-path', type=str, required=True)
    arg = arg_parser.parse_args()
    
    result_path = arg.result_path

    if not os.path.exists(result_path):
        print("No result directory exists under path: %s" % result_path)
        exit(0)

    df_aggreg = None

    count = 0
    for file in os.listdir(result_path):
        if file.endswith("xlsx") and not "best_params" in file:
            filepath = os.path.join(result_path, file)
            df_tpr = pd.read_excel(filepath, sheet_name="tpr_sorted", index_col=0)
            df_f1 = pd.read_excel(filepath, sheet_name="f1_sorted", index_col=0)
            count += 1

            if df_aggreg is None:
                df_aggreg = df_tpr[["sigma", "lambda", "sep_threshold", "precision", "tpr", "fpr", "f1"]].copy()
            else:
                df_aggreg[["precision", "tpr", "fpr", "f1"]] += df_tpr[["precision", "tpr", "fpr", "f1"]]

    if df_aggreg is not None:
        df_aggreg["points"] = df_aggreg["precision"] + df_aggreg["tpr"] + df_aggreg["f1"] \
                              + (1 - df_aggreg["fpr"])
        df_aggreg[["precision", "tpr", "fpr", "f1", "points"]] /= count


        df_aggreg["f1_rnd"] = np.around(df_aggreg["f1"] + 0.05, 1) \
                              - np.sign(np.around(df_aggreg["f1"] + 0.05, 1) - np.around(df_aggreg["f1"], 1)) * 0.05
        df_aggreg["tpr_rnd"] = np.around(df_aggreg["tpr"] + 0.05, 1) \
                               - np.sign(np.around(df_aggreg["tpr"] + 0.05, 1) - np.around(df_aggreg["tpr"], 1)) * 0.05
        df_aggreg["fpr_rnd"] = np.around(df_aggreg["fpr"] + 0.05, 1) \
                               - np.sign(np.around(df_aggreg["fpr"] + 0.05, 1) - np.around(df_aggreg["fpr"], 1)) * 0.05
        df_aggreg["prec_rnd"] = np.around(df_aggreg["precision"] + 0.05, 1) \
                                - np.sign(np.around(df_aggreg["precision"] + 0.05, 1) - np.around(df_aggreg["precision"], 1)) * 0.05


        best_params_path = os.path.join(result_path, "best_params.xlsx")
        with pd.ExcelWriter(best_params_path) as writer:
            df_tpr_metrics = df_aggreg.sort_values(by=["tpr_rnd", "f1_rnd", "fpr_rnd", "prec_rnd"], ascending=(False, False, True, False))
            df_f1_metrics = df_aggreg.sort_values(by=["f1_rnd", "tpr_rnd", "fpr_rnd", "prec_rnd"],
                                                     ascending=(False, False, True, False))
            df_points = df_aggreg.sort_values(by=["points"], ascending=False)

            df_tpr_metrics.to_excel(writer, sheet_name="tpr_metrics")
            df_f1_metrics.to_excel(writer, sheet_name="f1_metrics")
            df_points.to_excel(writer, sheet_name="points")
    else:
        print("Nothing to do! Bummer!")
