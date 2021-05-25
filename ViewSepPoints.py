from argparse import ArgumentParser
import os
import yaml
import pickle

from src.utils.WindowEventsParser import WindowEventsParser
from CPDStatisticsGenerator import apply_threshold, remove_consecutive_SEP_points


def load_configurations(config_file_path: str):
    with open(config_file_path) as config_file:
        return yaml.load(config_file)


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


def compute_positives(SEP, all_index_labels, match_interval, exclude_other=False):
    TP = []
    FP = []

    for kk, (sep_val, sep_index) in enumerate(SEP):
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
                            prev_sep_index = SEP[kk - 1][1] - 1
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
                        if kk < len(SEP) - 1:
                            next_sep_index = SEP[kk + 1][1] - 1
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
            TP.append((sep_index_in_file, sep_val))
        else:
            FP.append((sep_index_in_file, sep_val))

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


def compute_negatives(SEP, all_index_labels, match_interval, exclude_other):
    FN = []
    TN = []

    SEP_indexes = [tup[1] for tup in SEP]

    for index in range(1, len(all_index_labels)):
        current_index = all_index_labels[index][0]
        current_label = all_index_labels[index][1]

        previous_label = all_index_labels[index - 1][1]

        sep_in_range = is_sep_in_range(current_index - 1, SEP_indexes, match_interval)

        if not sep_in_range:
            if current_label != previous_label:
                FN.append(current_index - 1)
            else:
                TN.append(current_index - 1)
        # else:
        #     if current_label == previous_label:
        #         # if an event where there is no activity transition is caught in the sep_range of at least one
        #         # sep_index, then it is considered a false negative - because, had the event been a transition, it would
        #         # have been identified as a TP
        #         FN = FN + 1

    return TN, FN


def view_sep_statistics(output_file, SEP, all_events, match_interval, exclude_other):
    all_index_labels = [(ev.index, ev.label) for ev in all_events]
    # SEP_indexes = [tup[1] for tup in SEP]

    TP, FP = compute_positives(SEP, all_index_labels, match_interval, exclude_other)
    TN, FN = compute_negatives(SEP, all_index_labels, match_interval, exclude_other)

    TP_indexes = [tup[0] for tup in TP]
    FP_indexes = [tup[0] for tup in FP]

    with open(output_file, "w+") as out:
        for ev in all_events:
            out.write(ev.to_string())
            if ev.index in TP_indexes:
                out.write("\t")
                out.write("## TP: %5.2f" % TP[TP_indexes.index(ev.index)][1])
            elif ev.index in FP_indexes:
                out.write("\t")
                out.write("## FP: %5.2f" % FP[FP_indexes.index(ev.index)][1])
            elif ev.index in TN:
                out.write("\t")
                out.write("## TN")
            elif ev.index in FN:
                out.write("\t")
                out.write("## FN")

            out.write("\n")


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--config', type=str, required=True)

    arg_parser.add_argument('--lamda', type=float, required=True)
    arg_parser.add_argument('--sigma', type=float, required=True)

    arg_parser.add_argument('--sep-threshold', type=float, required=True, default=0.1)
    arg_parser.add_argument('--match-interval', type=int, required=True, default=5)

    arg = arg_parser.parse_args()

    config_name = os.path.splitext(os.path.basename(arg.config))[0]
    sep_threshold = arg.sep_threshold
    match_interval = arg.match_interval
    lamda = arg.lamda
    sigma = arg.sigma

    CONFIGURATIONS = load_configurations(arg.config)

    N = CONFIGURATIONS['N']
    CHANGEPOINT_WINDOW_STEP = CONFIGURATIONS['changepoint-window-step']
    FEATURES = CONFIGURATIONS['features']
    DATA_SET = CONFIGURATIONS['source-file']
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

    source_file_name = os.path.splitext(os.path.basename(DATA_SET))[0]
    dest_folder = "src" + os.path.sep + "results" + os.path.sep + config_name + os.path.sep + "N_%i_wlen_%i" \
                  % (N, WINDOW_LENGTH) + os.path.sep
    grid_search_folder = dest_folder + "grid_search" + os.path.sep

    res_file_name = grid_search_folder + source_file_name + "_res_%3.2f_%3.2f" % (sigma, lamda)
    pickled_sep_results_file = res_file_name + ".pkl"
    SEP = pickle.load(open(pickled_sep_results_file, "rb"))

    print("[INFO] Generating SEP stats view file for sigma: %5.2f, lambda: %5.2f, threshold: %5.2f, match_interval: %i"
          " ..." % (sigma, lamda, sep_threshold, match_interval))
    filtered_SEP = apply_threshold(SEP, sep_threshold)
    filtered_SEP = remove_consecutive_SEP_points(filtered_SEP)

    output_file = grid_search_folder + source_file_name + "_sep_view_%3.2f_%3.2f_%3.2f_%i.txt" \
                  % (sigma, lamda, sep_threshold, match_interval)
    view_sep_statistics(output_file, filtered_SEP, all_events, match_interval, exclude_other=EXCLUDE_OTHER_ACTIVITY)
