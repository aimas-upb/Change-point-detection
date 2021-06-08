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

    for file in os.listdir(result_path):
        if file.endswith("xlsx"):
            filepath = os.path.join(result_path, file)
            df = pd.read_excel(filepath, sheet_name="all_stats", index_col=0)
            if "f1_rnd" not in df.columns:
                df["f1_rnd"] = np.around(df["f1"] + 0.05, 1) \
                           - np.sign(np.around(df["f1"] + 0.05, 1) - np.around(df["f1"], 1)) * 0.05

            if "tpr_rnd" not in df.columns:
                df["tpr_rnd"] = np.around(df["tpr"] + 0.05, 1) \
                            - np.sign(np.around(df["tpr"] + 0.05, 1) - np.around(df["tpr"], 1)) * 0.05

            if "fpr_rnd" not in df.columns:
                df["fpr_rnd"] = np.around(df["fpr"] + 0.05, 1) \
                            - np.sign(np.around(df["fpr"] + 0.05, 1) - np.around(df["fpr"], 1)) * 0.05

            if "prec_rnd" not in df.columns:
                df["prec_rnd"] = np.around(df["precision"] + 0.05, 1) \
                             - np.sign(np.around(df["precision"] + 0.05, 1) - np.around(df["precision"], 1)) * 0.05

            df_tpr = df.sort_values(by=["tpr_rnd", "f1_rnd", "fpr_rnd", "prec_rnd"], ascending=(False, False, True, False))
            df_f1 = df.sort_values(by=["f1_rnd", "tpr_rnd", "fpr_rnd", "prec_rnd"], ascending=(False, False, True, False))

            rounded_stats_file = os.path.splitext(file)[0] + "_rnd" + ".xlsx"
            rounded_stats_dir = os.path.join(result_path, "rounded")
            if not os.path.exists(rounded_stats_dir):
                os.makedirs(rounded_stats_dir)

            rounded_stats_path = os.path.join(rounded_stats_dir, rounded_stats_file)
            with pd.ExcelWriter(rounded_stats_path) as writer:
                df_tpr.to_excel(writer, sheet_name="tpr_sorted")
                df_f1.to_excel(writer, sheet_name="f1_sorted")
