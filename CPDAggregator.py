import json
import os
import pickle
import timeit
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from densratio import densratio

from CPDStatisticsGenerator import apply_threshold, remove_consecutive_SEP_points
from RunGridSearch import compute_statistics, load_configurations
from SEPGenerator import build_features_from_config
from src.features.extractor.FeatureExtractor import FeatureExtractor
from src.models.Window import Window
from src.utils.WindowEventsParser import WindowEventsParser


def build_feature_window(all_events, window_length, data_set, feature_extractor):
    feature_windows = []

    source_file_name = os.path.splitext(os.path.basename(data_set))[0]
    dest_folder = "src" + os.path.sep + "results" + os.path.sep
    dest_file = dest_folder + source_file_name + ".pkl"

    if os.path.exists(dest_file):
        feature_windows = pickle.load(open(dest_file, "rb"))
    else:
        for index in range(0, len(all_events) - window_length + 1):
            print(index)
            # get current 30 events window
            window = Window(all_events[index:window_length + index])
            # get array of features from window
            feature_window = feature_extractor.extract_features_from_window(window)
            feature_windows.append(feature_window)

        pickle.dump(feature_windows, open(dest_file, "wb"))

    return feature_windows


def get_data_driven_SEP_points(feature_windows, N, kernel_param, regularization_param, window_length):
    SEP = []

    for index in range(N, len(feature_windows) + 1 - N):
        print('Index SEP: ' + str(index) + '/' + str(len(feature_windows) + 1 - N))
        previous_x = feature_windows[index - N: index]
        assert len(previous_x) == N

        current_x = feature_windows[index: N + index]
        assert len(current_x) == N

        # use previous_x as the Y samples for distribution of f_(t-1)(x) and
        # use current_x as the X samples for distribution f_t(x) in a call to densratio RuLSIF -
        densratio_res = densratio(x=np.array(current_x), y=np.array(previous_x), kernel_num=len(previous_x),
                                  sigma_range=[kernel_param], lambda_range=[regularization_param],
                                  verbose=False)

        g_sum = np.sum(densratio_res.compute_density_ratio(np.array(current_x))) / len(current_x)
        sep = max(0, 0.5 - g_sum)
        print(g_sum)

        sensor_index = index - 1 + window_length
        SEP.append((sep, sensor_index))

    return SEP


def get_knowledge_driven_SEP_points(KN_results_location):
    input_file = open(KN_results_location)
    json_array = json.load(input_file)
    KD_SEP = []

    for index in json_array["indexes"]:
        CP_index = index.split()[1]
        KD_SEP.append((0.5, int(CP_index)))

    return KD_SEP


def combine_approaches(DD_SEP, KD_SEP):
    KD_SEP_indexes = [tup[1] for tup in KD_SEP]
    SEP = KD_SEP

    for dd_SEP in DD_SEP:
        if dd_SEP[1] not in KD_SEP_indexes and \
                dd_SEP[1] + 1 not in KD_SEP_indexes and \
                dd_SEP[1] - 1 not in KD_SEP_indexes and \
                dd_SEP[1] + 2 not in KD_SEP_indexes and \
                dd_SEP[1] - 2 not in KD_SEP_indexes:
            SEP.append(dd_SEP)
    return sorted(SEP, key=lambda tup: tup[1])


def compute_performance_results(MAX_CP_MATCH_INTERVAL, SEP, THRESHOLD, EXCLUDE_OTHER_ACTIVITY, KERNEL_PARAM, REGULARIZATION_PARAM):
    all_stats = []

    # filter SEP data and produce DataFrame with performance metrics
    for match_interval in range(0, MAX_CP_MATCH_INTERVAL + 1):
        stats_dict = compute_statistics(SEP, all_events, match_interval,
                                        EXCLUDE_OTHER_ACTIVITY)

        all_stats_dict = {
            "sigma": KERNEL_PARAM,
            "lambda": REGULARIZATION_PARAM,
            "sep_threshold": THRESHOLD,
            "match_interval": match_interval
        }
        all_stats_dict.update(stats_dict)
        all_stats.append(all_stats_dict)

    return all_stats


def save_performance_results(all_stats):
    config_name = os.path.splitext(os.path.basename(arg.config))[0]
    source_file_name = os.path.splitext(os.path.basename(DATA_SET))[0]

    results_folder = 'src/results/final-results/'

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    all_stats_df = pd.DataFrame(all_stats)
    all_stats_df = all_stats_df.sort_values(by=["f1", "recall", "precision"], ascending=False)
    all_stats_file = results_folder + source_file_name + "_" + config_name + "_all_stats" + ".xlsx"

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


if __name__ == "__main__":
    start = timeit.default_timer()

    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--config', type=str, required=True)

    arg = arg_parser.parse_args()
    CONFIGURATIONS = load_configurations(arg.config)

    WINDOW_LENGTH = CONFIGURATIONS['window-length']
    DATA_SET = CONFIGURATIONS['data-set']
    N = CONFIGURATIONS['N']
    THRESHOLD = CONFIGURATIONS['threshold']
    KERNEL_PARAM = CONFIGURATIONS['kernel-param']
    REGULARIZATION_PARAM = CONFIGURATIONS['regularization-param']
    EXCLUDE_OTHER_ACTIVITY = CONFIGURATIONS['exclude-other']
    MAX_CP_MATCH_INTERVAL = CONFIGURATIONS['max-cp-match-interval']
    FEATURES = CONFIGURATIONS['features']
    KN_RESULTS_LOCATION = CONFIGURATIONS['kn-results-location']
    ONLY_DATA_DRIVEN = CONFIGURATIONS['only-DD']
    ONLY_KNOWLEDGE_DRIVEN = CONFIGURATIONS['only-KD']
    DATA_DRIVEN_AND_KNOWLEDGE_DRIVEN = CONFIGURATIONS['DD-and-KD']

    parser = WindowEventsParser()
    parser.read_data_from_file(DATA_SET)
    all_events = parser.events

    DD_SEP = []
    KD_SEP = []
    SEP = []

    if ONLY_DATA_DRIVEN or DATA_DRIVEN_AND_KNOWLEDGE_DRIVEN:
        features = build_features_from_config(FEATURES)
        feature_extractor = FeatureExtractor(features)
        feature_windows = build_feature_window(all_events, WINDOW_LENGTH, DATA_SET, feature_extractor)

        # gets list of SEP points in data driven approach
        DD_SEP = get_data_driven_SEP_points(feature_windows, N, KERNEL_PARAM, REGULARIZATION_PARAM, WINDOW_LENGTH)
        DD_SEP = apply_threshold(DD_SEP, THRESHOLD)
        DD_SEP = remove_consecutive_SEP_points(DD_SEP)
        SEP = DD_SEP

    if ONLY_KNOWLEDGE_DRIVEN or DATA_DRIVEN_AND_KNOWLEDGE_DRIVEN:
        # gets list of SEP points in knowledge driven approach
        KD_SEP = get_knowledge_driven_SEP_points(KN_RESULTS_LOCATION)
        SEP = KD_SEP

    if DATA_DRIVEN_AND_KNOWLEDGE_DRIVEN:
        # combines the two approaches
        SEP = combine_approaches(DD_SEP, KD_SEP)

    # compute performance statistics
    PERFORMANCE_RESULTS = compute_performance_results(MAX_CP_MATCH_INTERVAL, SEP, THRESHOLD, EXCLUDE_OTHER_ACTIVITY, KERNEL_PARAM, REGULARIZATION_PARAM)
    # saves all statistics to file
    save_performance_results(PERFORMANCE_RESULTS)