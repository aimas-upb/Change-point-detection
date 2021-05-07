import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import yaml
from densratio import densratio

from CPDStatisticsGenerator import apply_threshold, remove_consecutive_SEP_points
from SEPGenerator import build_features_from_config, save_sep_data, add_sep_assignment
from src.features.extractor.FeatureExtractor import FeatureExtractor
from src.models.Window import Window
from src.utils.Encoder import Encoder
from src.utils.WindowEventsParser import WindowEventsParser

OTHER_ACTIVITY = "Other_Activity"


def get_sep_index_neighbours(sep_index_in_file, all_index_labels, match_interval):
    sep_index_neighbours = []

    for index in range(-match_interval, 0):
        if sep_index_in_file + index >= 0:
            sep_index_neighbours.append(all_index_labels[sep_index_in_file + index])

    for index in range(match_interval + 1):
        if sep_index_in_file + index < len(all_index_labels):
            sep_index_neighbours.append(all_index_labels[sep_index_in_file + index])

    # sep_index_neighbours.sort(key=lambda x: x[0])
    # print('SEP: ' + str(sep_index_in_file + 1) + ' -> sep_index_neighbours: ' + str(sep_index_neighbours))
    return sep_index_neighbours


def compute_positives(SEP_indexes, all_index_labels, match_interval, exclude_other):
    TP = 0
    FP = 0

    for sep_index in SEP_indexes:
        sep_index_in_file = sep_index - 1
        sep_index_neighbours = get_sep_index_neighbours(sep_index_in_file, all_index_labels, match_interval)

        is_at_least_one_transition = False

        for index in range(1, len(sep_index_neighbours)):
            if exclude_other:
                if sep_index_neighbours[index][1] != OTHER_ACTIVITY and sep_index_neighbours[index - 1][
                    1] != OTHER_ACTIVITY \
                        and sep_index_neighbours[index][1] != sep_index_neighbours[index - 1][1]:
                    # print(sep_index_neighbours[index])
                    is_at_least_one_transition = True
                    break
            else:
                if sep_index_neighbours[index][1] != sep_index_neighbours[index - 1][1]:
                    # print(sep_index_neighbours[index])
                    is_at_least_one_transition = True
                    break

        if is_at_least_one_transition:
            TP = TP + 1
        else:
            FP = FP + 1

    return TP, FP


def is_sep_in_range(current_index, SEP_indexes, match_interval):
    for i, sep_idx in enumerate(SEP_indexes):
        if sep_idx >= current_index:
            if sep_idx <= current_index + match_interval:
                return True
            elif i > 0 and SEP_indexes[i - 1] >= current_index - match_interval:
                return True
            else:
                return False

    if SEP_indexes and SEP_indexes[-1] >= current_index - match_interval:
        return True

    return False


def compute_negatives(SEP_indexes, all_index_labels, match_interval, exclude_other):
    FN = 0
    TN = 0

    for index in range(1, len(all_index_labels)):
        current_index = all_index_labels[index][0]
        current_label = all_index_labels[index][1]

        previous_label = all_index_labels[index - 1][1]

        sep_in_range = is_sep_in_range(current_index, SEP_indexes, match_interval)

        if not sep_in_range:
            if exclude_other:
                if current_label != OTHER_ACTIVITY and previous_label != OTHER_ACTIVITY and current_label != previous_label:
                    FN = FN + 1
                else:
                    TN = TN + 1
            else:
                if current_label != previous_label:
                    FN = FN + 1
                else:
                    TN = TN + 1

    return TN, FN


def count_statistics(SEP, all_index_labels, match_interval, exclude_other):
    SEP_indexes = [tup[1] for tup in SEP]

    TP, FP = compute_positives(SEP_indexes, all_index_labels, match_interval, exclude_other)
    TN, FN = compute_negatives(SEP_indexes, all_index_labels, match_interval, exclude_other)

    return FN, FP, TN, TP


def compute_statistics(SEP, all_events, match_interval, exclude_other):
    all_index_labels = [(ev.index, ev.label) for ev in all_events]
    FN, FP, TN, TP = count_statistics(SEP, all_index_labels, match_interval, exclude_other)

    precision = 0
    if TP + FP != 0:
        precision = TP / (TP + FP)

    recall = 0
    if TP + FN != 0:
        recall = TP / (TP + FN)

    accuracy = 0
    if TP + FP + TN + FN != 0:
        accuracy = (TP + TN) / (TP + FP + TN + FN)

    f1 = 0
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "tp": TP,
        "fp": FP,
        "tn": TN,
        "fn": FN,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1
    }


def load_configurations(config_file_path: str):
    with open(config_file_path) as config_file:
        return yaml.load(config_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--config', type=str, required=True)
    arg = arg_parser.parse_args()

    CONFIGURATIONS = load_configurations(arg.config)

    N = CONFIGURATIONS['N']
    FEATURES = CONFIGURATIONS['features']
    DATA_SET = CONFIGURATIONS['source-file']
    WINDOW_LENGTH = CONFIGURATIONS['window-length']
    THRESHOLD_STEP = CONFIGURATIONS['threshold']['step']
    MAX_SEP_THRESHOLD = CONFIGURATIONS['threshold']['max']
    KERNEL_PARAM_GRID = CONFIGURATIONS['kernel-param-grid']
    EXCLUDE_OTHER_ACTIVITY = CONFIGURATIONS['exclude-other']
    MAX_CP_MATCH_INTERVAL = CONFIGURATIONS['max-cp-match-interval']
    REGULARIZATION_PARAM_GRID = CONFIGURATIONS['regularization-param-grid']

    # data parser
    parser = WindowEventsParser()
    parser.read_data_from_file(DATA_SET)
    all_events = parser.events

    # features
    # defines the list of features that will be extracted from each window
    features = build_features_from_config(FEATURES)
    feature_extractor = FeatureExtractor(features)

    feature_windows = []
    oneHotEncoder = Encoder()

    source_file_name = os.path.splitext(os.path.basename(DATA_SET))[0]
    dest_folder = "src" + os.path.sep + "results" + os.path.sep + "N_%i_wlen_%i" % (N, WINDOW_LENGTH) + os.path.sep

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    dest_file = dest_folder + source_file_name + ".pkl"

    if os.path.exists(dest_file):
        feature_windows = pickle.load(open(dest_file, "rb"))
    else:
        for i in range(0, len(all_events) - WINDOW_LENGTH + 1):
            # get current 30 events window
            window = Window(all_events[i:WINDOW_LENGTH + i])
            # get array of features from window
            feature_window = feature_extractor.extract_features_from_window(window)
            feature_windows.append(feature_window)

        pickle.dump(feature_windows, open(dest_file, "wb"))

    grid_search_folder = dest_folder + "grid_search" + os.path.sep
    if not os.path.exists(grid_search_folder):
        os.makedirs(grid_search_folder)

    all_stats = []

    for sigma in KERNEL_PARAM_GRID:
        for lamda in REGULARIZATION_PARAM_GRID:

            print("[INFO] Computing stats for sigma: %5.2f, lambda: %5.2f ..." % (sigma, lamda))

            res_file_name = grid_search_folder + source_file_name + "_res_%3.2f_%3.2f" % (sigma, lamda)

            SEP = []
            SEP_assignments = []

            for index in range(N, len(feature_windows) + 1 - N):
                print('Index SEP: ' + str(index) + '/' + str(len(feature_windows) + 1 - N))
                previous_x = feature_windows[index - N: index]
                assert len(previous_x) == N

                current_x = feature_windows[index: N + index]
                assert len(current_x) == N

                # use previous_x as the Y samples for distribution of f_(t-1)(x) and
                # use current_x as the X samples for distribution f_t(x) in a call to densratio RuLSIF -
                densratio_res = densratio(x=np.array(current_x), y=np.array(previous_x), kernel_num=len(previous_x),
                                          sigma_range=[sigma], lambda_range=[lamda],
                                          verbose=False)

                g_sum = np.sum(densratio_res.compute_density_ratio(np.array(current_x))) / len(current_x)
                sep = max(0, 0.5 - g_sum)

                # sensor_index = feature_windows.index(previous_x[N - 1]) + WINDOW_LENGTH
                sensor_index = index - 1 + WINDOW_LENGTH

                add_sep_assignment(sensor_index, sep, all_events, SEP_assignments, feature_extractor, WINDOW_LENGTH)

                SEP.append((round(sep, 4), sensor_index))

            save_sep_data(res_file_name, SEP, SEP_assignments)

            # filter SEP data and produce DataFrame with performance metrics
            for match_interval in range(MAX_CP_MATCH_INTERVAL):
                for sep_threshold in np.arange(THRESHOLD_STEP, MAX_SEP_THRESHOLD, THRESHOLD_STEP):
                    # for match_interval in [1, 2]:
                    #     for sep_threshold in [0.05, 0.1]:

                    filtered_SEP = apply_threshold(SEP, sep_threshold)
                    filtered_SEP = remove_consecutive_SEP_points(filtered_SEP)

                    stats_dict = compute_statistics(filtered_SEP, all_events, match_interval,
                                                    EXCLUDE_OTHER_ACTIVITY)

                    all_stats_dict = {
                        "sigma": sigma,
                        "lambda": lamda,
                        "sep_threshold": sep_threshold,
                        "match_interval": match_interval
                    }
                    all_stats_dict.update(stats_dict)
                    all_stats.append(all_stats_dict)

    all_stats_df = pd.DataFrame(all_stats)
    all_stats_df = all_stats_df.sort_values(by=["f1", "recall", "precision"], ascending=False)
    all_stats_file = grid_search_folder + source_file_name + "_all_stats" + ".xlsx"

    max_precision_idx = all_stats_df["precision"].idxmax()
    max_recall_idx = all_stats_df["recall"].idxmax()
    max_f1_idx = all_stats_df["f1"].idxmax()
    max_accuracy_idx = all_stats_df["accuracy"].idxmax()

    max_df = pd.DataFrame(data=[], columns=list(all_stats_df.columns.values))
    max_df = max_df.append(all_stats_df.iloc[max_precision_idx])
    max_df = max_df.append(all_stats_df.iloc[max_recall_idx])
    max_df = max_df.append(all_stats_df.iloc[max_f1_idx])
    max_df = max_df.append(all_stats_df.iloc[max_accuracy_idx])

    with pd.ExcelWriter(all_stats_file) as writer:
        all_stats_df.to_excel(writer, sheet_name="all_stats")
        max_df.to_excel(writer, sheet_name="max_stats")
