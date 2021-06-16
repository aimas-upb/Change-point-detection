import pickle
import timeit
from argparse import ArgumentParser

from src.models.StatisticsData import StatisticsData
from src.models.TransitionAccuracy import TransitionAccuracy

OTHER_ACTIVITY = "Other_Activity"


def compute_statistics(SEP):
    FN, FP, TN, TP = count_statistics(SEP)
    print_and_save_statistics(FN, FP, TN, TP)


def count_statistics(SEP):
    SEP_indexes = [tup[1] for tup in SEP]

    TP, FP = compute_positives(SEP_indexes)
    TN, FN = compute_negatives(SEP_indexes)

    return FN, FP, TN, TP


def compute_positives(SEP_indexes):
    TP = 0
    FP = 0

    for sep_index in SEP_indexes:
        sep_index_in_file = sep_index - 1
        sep_index_neighbours = get_sep_index_neighbours(sep_index_in_file)

        is_at_least_one_transition = False

        for index in range(1, len(sep_index_neighbours)):
            if EXCLUDE_OTHER_ACTIVITY_FROM_STATISTICS:
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


def compute_negatives(SEP_indexes):
    FN = 0
    TN = 0

    for index in range(1, len(all_index_labels)):
        current_index = all_index_labels[index][0]
        current_label = all_index_labels[index][1]

        previous_label = all_index_labels[index - 1][1]

        if current_index not in SEP_indexes:
            if EXCLUDE_OTHER_ACTIVITY_FROM_STATISTICS:
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


def print_and_save_statistics(FN, FP, TN, TP):
    # print("===============")
    # print("TP: " + str(TP))
    # print("TN: " + str(TN))
    # print("FP: " + str(FP))
    # print("FN: " + str(FN))

    precision = compute_precision(FP, TP)
    recall = compute_recall(FN, TP)
    accuracy = compute_accuracy(TP, TN, FP, FN)

    # print("==================")
    # print("PRECISION: " + str(precision))
    # print("RECALL: " + str(recall))
    # print("ACCURACY: " + str(accuracy))
    # print("==================")

    print("THRESHOLD: " + str(THRESHOLD))
    print("CP_MATCH_INTERVAL: " + str(CP_MATCH_INTERVAL))

    STATISTICS_DATA.append(StatisticsData(precision, recall, accuracy, THRESHOLD, CP_MATCH_INTERVAL))


def compute_recall(false_negative, true_positive):
    if true_positive + false_negative != 0:
        return round(true_positive / (true_positive + false_negative), 2)
    return 0.0


def compute_precision(false_positive, true_positive):
    if true_positive + false_positive != 0:
        return round(true_positive / (true_positive + false_positive), 2)
    return 0.0


def compute_accuracy(true_positive, true_negative, false_positive, false_negative):
    if true_positive + true_negative + false_negative + false_positive != 0:
        return round(
            (true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive), 2)


def write_statistics_to_file():
    f = open("statistics_data.txt", "a")

    for statistic_data in STATISTICS_DATA:
        f.write('MATCH INTERVAL: ' + str(statistic_data.match_interval) + ' - THRESHOLD: ' + str(
            round(statistic_data.threshold, 2)) +
                ' - PRECISION: ' + str(statistic_data.precision) + ' - RECALL: ' + str(
            statistic_data.recall) + ' - ACCURACY: ' + str(statistic_data.accuracy) + '\n')
    f.close()


def load_pickle(file: str):
    with open(file, 'rb') as file:
        return pickle.load(file)


def apply_threshold(SEP, THRESHOLD):
    return [sep for sep in SEP if sep[0] > THRESHOLD]


def remove_consecutive_SEP_points(SEP):
    if len(SEP) == 0:
        return []

    non_consecutive = []
    consecutive = []

    print("a")

    for sep in SEP:
        if len(consecutive) == 0:
            consecutive.append(sep)
        else:
            if sep[1] - 10 <= consecutive[len(consecutive) - 1][1]:
                consecutive.append(sep)
            else:
                non_consecutive.append(max(consecutive, key=lambda k: k[0]))
                consecutive = [sep]

    if len(consecutive) > 0:
        non_consecutive.append(max(consecutive))

    return non_consecutive


def contains_transition(sensor_labels_interval):
    if len(sensor_labels_interval) == 1:
        sensor_index = all_index_labels.index(sensor_labels_interval[0])
        if sensor_index > 0 and all_index_labels[sensor_index - 1][1] != all_index_labels[sensor_index][1]:
            print(str(all_index_labels[sensor_index - 1][1]) + " to " + str(all_index_labels[sensor_index][1]))
            return True

    for index in range(1, len(sensor_labels_interval)):
        if sensor_labels_interval[index][1] != sensor_labels_interval[index - 1][1]:
            return True
    return False


def is_SEP_for_index(index: int, SEP_indexes):
    return index in SEP_indexes


def is_transition(index, all_index_labels):
    print(str(all_index_labels[index - 1][1]) + " != " + str(all_index_labels[index][1]) + " == " + str(
        all_index_labels[index - 1][1] != all_index_labels[index][1]))
    return all_index_labels[index - 1][1] != all_index_labels[index][1]


def get_sep_index_neighbours(sep_index_in_file: int):
    sep_index_neighbours = [all_index_labels[sep_index_in_file - CP_MATCH_INTERVAL - 1]]

    for index in range(0, CP_MATCH_INTERVAL + 1):
        sep_index_neighbours.append(all_index_labels[sep_index_in_file + index])

        if index > 0:
            sep_index_neighbours.append(all_index_labels[sep_index_in_file - index])

    sep_index_neighbours.sort(key=lambda x: x[0])
    # print('SEP: ' + str(sep_index_in_file + 1) + ' -> sep_index_neighbours: ' + str(sep_index_neighbours))

    return sep_index_neighbours


def compute_transitions_for_activities(activities):
    labels = set([tup[1] for tup in activities])

    dicts = []
    for label in labels:
        dicts.append({label: {}})

    for activity_index in range(1, len(activities)):
        d = {}
        activity = activities[activity_index][1]
        last_activity = activities[activity_index - 1][1]

        if any(item for item in dicts if last_activity in item):
            d = next(item for item in dicts if last_activity in item)

        if len(d) > 0:
            if activity in d[last_activity]:
                if activity == last_activity:
                    d[last_activity][activity] = d[last_activity][activity] + 1
                else:
                    d[last_activity][activity].append(activities[activity_index][0])
            else:
                if activity == last_activity:
                    d[last_activity][activity] = 1
                else:
                    d[last_activity][activity] = [activities[activity_index][0]]
        else:
            dicts.append({last_activity: {}})

    return dicts


def get_avg_acc_for_activity(results, activity: str, main: bool):
    activity_accuracies = []

    for result in results:
        if main:
            if result.main_activity == activity:
                activity_accuracies.append(result.accuracy)
        else:
            if result.transition_activity == activity:
                activity_accuracies.append(result.accuracy)

    return round(sum(activity_accuracies) / len(activity_accuracies), 2)


def count_number_of_sep_match(SEP, all_index_labels, CP_MATCH_INTERVAL):
    match = 0
    for SEP, index in SEP:
        if index <= CP_MATCH_INTERVAL and contains_transition(all_index_labels[index - 1: index + CP_MATCH_INTERVAL]) or \
                index >= SEP_size - CP_MATCH_INTERVAL and contains_transition(
            all_index_labels[index - CP_MATCH_INTERVAL - 1: index]):
            match = match + 1
        else:
            if contains_transition(all_index_labels[index - CP_MATCH_INTERVAL - 1: index + CP_MATCH_INTERVAL]):
                match = match + 1
    print('Accuracy: ' + str(match * 100 / SEP_size) + ' (' + str(match) + '/' + str(SEP_size) + ')')


def compute_from_and_to_activity_match(all_index_labels, SEP, CP_MATCH_INTERVAL):
    all_dicts = compute_transitions_for_activities(all_index_labels)
    SEP_indexes = [tup[1] for tup in SEP]
    results = []

    for activity_dict in all_dicts:
        main_activity = list(activity_dict.keys())[0]

        for transition_dict in activity_dict:
            inner_transition_dict = activity_dict[transition_dict]

            for activity in inner_transition_dict:
                transitions_values = inner_transition_dict[activity]

                match = 0
                accuracy = 0

                if isinstance(transitions_values, list):
                    for transition_index in transitions_values:
                        acceptance_interval = list(
                            range(transition_index - CP_MATCH_INTERVAL, transition_index + CP_MATCH_INTERVAL + 1))
                        for index in acceptance_interval:
                            if index in SEP_indexes:
                                match = match + 1
                                break

                    accuracy = match / len(transitions_values) * 100

                # print(str(activity) + " -> " + str(transitions_values) + " -> ACCURACY: " + str(accuracy))
                result = TransitionAccuracy(main_activity, activity, accuracy)

                if main_activity != activity:
                    results.append(result)

    return results


if __name__ == "__main__":
    start = timeit.default_timer()

    arg_parser = ArgumentParser(description='.')

    # data parser
    arg = arg_parser.parse_args()

    SEP = load_pickle('./src/results/hh103-with-locations_res_1.00_0.50.pkl')
    all_index_labels = load_pickle('./pickles/index-labels/hh103-index-label.pkl')

    EXCLUDE_OTHER_ACTIVITY_FROM_STATISTICS = False
    STATISTICS_DATA = []

    # this method is only used for sep files that used partial list of seps
    # SEP = extend_sep(SEP)
    SEP_size = len(SEP)
    CP_MATCH_INTERVAL = 0

    # this method computes precision, recall, accuracy
    while CP_MATCH_INTERVAL < 10:
        THRESHOLD = 0.01

        while THRESHOLD < 0.5:
            # print('Initial size: ' + str(len(SEP)))
            filtered_SEP = apply_threshold(SEP, THRESHOLD)
            # print('After threshold: ' + str(len(SEP)))
            filtered_SEP = remove_consecutive_SEP_points(filtered_SEP)
            # print('Without consecutive CPs: ' + str(len(SEP)))

            compute_statistics(filtered_SEP)
            THRESHOLD = THRESHOLD + 0.01
            filtered_SEP = []

        CP_MATCH_INTERVAL = CP_MATCH_INTERVAL + 1

    write_statistics_to_file()

    # this method counts how many sep points are really transitions in data set
    # in can take into consideration any CP_MATCH_INTERVAL > 0
    # count_number_of_sep_match(SEP, all_index_labels, CP_MATCH_INTERVAL)

    # computes report of matches between SEP point and actual transition from an activity and to that activity
    # results = compute_from_and_to_activity_match(all_index_labels, SEP, CP_MATCH_INTERVAL)
    # writeResultsInExcelFile(results, 'example.xls')
    # print(sum([result.accuracy for result in results]) / len(results))
