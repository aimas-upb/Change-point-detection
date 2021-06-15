import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import yaml
from densratio import densratio

from CPDStatisticsGenerator import apply_threshold, remove_consecutive_SEP_points
from SEPGenerator import build_features_from_config, save_sep_data, add_sep_assignment
from src.features.base.Feature import Feature
from src.features.extractor.FeatureExtractor import FeatureExtractor
from src.models.Window import Window
from src.utils.Encoder import Encoder
from src.utils.WindowEventsParser import WindowEventsParser

OTHER_ACTIVITY = "Other_Activity"


def get_sep_index_neighbours(sep_index_in_file, all_index_labels, match_interval):
    sep_index_neighbours = []

    for index in range(-match_interval, 0):
        if 0 <= sep_index_in_file + index < len(all_index_labels):
            sep_index_neighbours.append(all_index_labels[sep_index_in_file + index])

    for index in range(match_interval + 1):
        if sep_index_in_file + index < len(all_index_labels):
            sep_index_neighbours.append(all_index_labels[sep_index_in_file + index])

    # sep_index_neighbours.sort(key=lambda x: x[0])
    # print('SEP: ' + str(sep_index_in_file + 1) + ' -> sep_index_neighbours: ' + str(sep_index_neighbours))
    return sep_index_neighbours


def compute_positives(SEP_indexes, all_events, match_interval, exclude_other=False):
    all_index_labels = [(ev.index, ev.label) for ev in all_events]
    TP = 0
    FP = 0

    for kk, sep_index in enumerate(SEP_indexes):
        sep_index_in_file = sep_index - 1
        sep_index_neighbours = get_sep_index_neighbours(sep_index_in_file, all_index_labels, match_interval)

        is_at_least_one_transition = False

        if match_interval == 0:
            if sep_index_in_file < len(all_index_labels) - 1:
                if all_index_labels[sep_index_in_file][1] != all_index_labels[sep_index_in_file + 1][1]:
                    is_at_least_one_transition = True
        else:
            for i in range(1, len(sep_index_neighbours)):
                if sep_index_neighbours[i][1] != sep_index_neighbours[i - 1][1]:
                    # there is at least one activity transition in the ground truth
                    event_index = sep_index_neighbours[i - 1][0]
                    current_sep_index = sep_index_neighbours[match_interval][0]

                    # we want to avoid double counting of TP given the match_interval
                    # so, we only count the current_sep_index as a TP if an activity change is within
                    # its match_interval AND there is no previous / next sep index that is CLOSER to that
                    # activity change
                    if i < match_interval + 1:
                        if kk > 1:
                            prev_sep_index = SEP_indexes[kk - 1] - 1
                            if prev_sep_index <= current_sep_index - 2 * match_interval:
                                is_at_least_one_transition = True
                                break
                            elif abs(current_sep_index - event_index) <= abs(prev_sep_index - event_index):
                                is_at_least_one_transition = True
                                break
                        else:
                            is_at_least_one_transition = True
                            break
                    else:
                        if kk < len(SEP_indexes) - 1:
                            next_sep_index = SEP_indexes[kk + 1] - 1
                            if next_sep_index >= current_sep_index + 2 * match_interval:
                                is_at_least_one_transition = True
                                break
                            elif abs(current_sep_index - event_index) < abs(next_sep_index - event_index):
                                is_at_least_one_transition = True
                                break
                        else:
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


def get_activity_slice(all_events, start_index, activity_label, direction = 1):
    idx = start_index
    while 0 <= idx and idx < len(all_events) and all_events[idx].label == activity_label:
        idx += direction

    return all_events[min(start_index, idx) : max(start_index, idx)]

def num_activations_in_event_slice(event_slice):
    activations = 0
    for ev in event_slice:
        if Feature.is_motion_sensor(ev.sensor.name):
            if ev.sensor.state == "ON":
                activations += 1
        else:
            activations += 1

    return activations


def compute_negatives(SEP_indexes, all_events, match_interval, exclude_other):
    FN = 0
    TN = 0

    index = 0
    while index < len(all_events) - 1:
        index += 1

        current_index = all_events[index].index
        current_label = all_events[index].label

        previous_label = all_events[index - 1].label

        sep_in_range = is_sep_in_range(current_index - 1, SEP_indexes, match_interval)

        if not sep_in_range:
            if current_label != previous_label:
                if exclude_other and current_label == OTHER_ACTIVITY:
                    continuous_other = get_activity_slice(all_events, index, OTHER_ACTIVITY)
                    if num_activations_in_event_slice(continuous_other) <= 2:
                        index += len(continuous_other)
                else:
                    FN = FN + 1
            else:
                TN = TN + 1
        # else:
        #     if current_label == previous_label:
        #         # if an event where there is no activity transition is caught in the sep_range of at least one
        #         # sep_index, then it is considered a false negative - because, had the event been a transition, it would
        #         # have been identified as a TP
        #         FN = FN + 1

    return TN, FN


def count_statistics(SEP, all_events, match_interval, exclude_other):
    SEP_indexes = [tup[1] for tup in SEP]

    TP, FP = compute_positives(SEP_indexes, all_events, match_interval, exclude_other)
    TN, FN = compute_negatives(SEP_indexes, all_events, match_interval, exclude_other)

    return FN, FP, TN, TP


def compute_statistics(SEP, all_events, match_interval, exclude_other):
    FN, FP, TN, TP = count_statistics(SEP, all_events, match_interval, exclude_other)

    precision = 0
    if TP + FP != 0:
        precision = TP / (TP + FP)

    recall = 0
    if TP + FN != 0:
        recall = TP / (TP + FN)

    tpr = recall

    fpr = 0
    if TN + FP != 0:
        fpr = FP / (TN + FP)

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
        "tpr": tpr,
        "fpr": fpr,
        "accuracy": accuracy,
        "f1": f1
    }


def load_configurations(config_file_path: str):
    with open(config_file_path) as config_file:
        return yaml.load(config_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--config', type=str, required=True)
    arg_parser.add_argument('--stats-only', type=str, required=False, default=False)
    arg_parser.add_argument('--save-sep', type=str, required=False, default=True)
    arg_parser.add_argument('--src', type=str, required=False)
    arg = arg_parser.parse_args()
    
    config_name = os.path.splitext(os.path.basename(arg.config))[0]
    stats_only = arg.stats_only
    save_sep = arg.save_sep
    src = arg.src
    
    CONFIGURATIONS = load_configurations(arg.config)

    N = CONFIGURATIONS['N']
    CHANGEPOINT_WINDOW_STEP = CONFIGURATIONS['changepoint-window-step']
    FEATURES = CONFIGURATIONS['features']
    DATA_SET = CONFIGURATIONS['source-file']
    if src:
        DATA_SET = src
    
    WINDOW_LENGTH = CONFIGURATIONS['window-length']
    THRESHOLD_STEP = CONFIGURATIONS['threshold']['step']
    MAX_SEP_THRESHOLD = CONFIGURATIONS['threshold']['max']
    MIN_SEP_THRESHOLD = CONFIGURATIONS['threshold']['min']
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
    dest_folder = "src" + os.path.sep + "results" + os.path.sep + config_name + os.path.sep + "N_%i_wlen_%i" \
                  % (N, WINDOW_LENGTH) + os.path.sep

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

    if not stats_only:
        for sigma in KERNEL_PARAM_GRID:
            for lamda in REGULARIZATION_PARAM_GRID:

                print("[INFO] Computing stats for sigma: %5.2f, lambda: %5.2f ..." % (sigma, lamda))

                res_file_name = grid_search_folder + source_file_name + "_res_%3.2f_%3.2f" % (sigma, lamda)

                SEP = []
                SEP_assignments = []

                for index in range(N + CHANGEPOINT_WINDOW_STEP, len(feature_windows) + 1 - N):
                    # print('Index SEP: ' + str(index) + '/' + str(len(feature_windows) + 1 - N))
                    previous_x = feature_windows[index - N - CHANGEPOINT_WINDOW_STEP: index - CHANGEPOINT_WINDOW_STEP]
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
                    sensor_index = index - 1 + WINDOW_LENGTH - CHANGEPOINT_WINDOW_STEP

                    add_sep_assignment(sensor_index, sep, all_events, SEP_assignments, feature_extractor, WINDOW_LENGTH)

                    SEP.append((sep, sensor_index))
                
                if save_sep:
                    save_sep_data(res_file_name, SEP, SEP_assignments)

                # filter SEP data and produce DataFrame with performance metrics
                for match_interval in range(MAX_CP_MATCH_INTERVAL - 1, MAX_CP_MATCH_INTERVAL):
                    for sep_threshold in np.arange(MIN_SEP_THRESHOLD, MAX_SEP_THRESHOLD, THRESHOLD_STEP):
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
    else:
        for sigma in KERNEL_PARAM_GRID:
            for lamda in REGULARIZATION_PARAM_GRID:
                res_file_name = grid_search_folder + source_file_name + "_res_%3.2f_%3.2f" % (sigma, lamda)
                pickled_sep_results_file = res_file_name + ".pkl"

                if os.path.exists(pickled_sep_results_file):
                    print("[INFO] Computing stats for sigma: %5.2f, lambda: %5.2f ..." % (sigma, lamda))
                    SEP = pickle.load(open(pickled_sep_results_file, "rb"))

                    # filter SEP data and produce DataFrame with performance metrics
                    for match_interval in range(MAX_CP_MATCH_INTERVAL):
                        for sep_threshold in np.arange(MIN_SEP_THRESHOLD, MAX_SEP_THRESHOLD, THRESHOLD_STEP):
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
    all_stats_file = grid_search_folder + source_file_name + "_" + config_name + "_all_stats" + ".xlsx"

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
