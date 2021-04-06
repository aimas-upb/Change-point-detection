import pickle
import timeit
from argparse import ArgumentParser

from src.models.TransitionAccuracy import TransitionAccuracy


def load_pickle(file: str):
    with open(file, 'rb') as file:
        return pickle.load(file)


def apply_threshold(SEP, THRESHOLD):
    return [sep for sep in SEP if sep[0] > THRESHOLD]


def remove_consecutive_SEP_points(SEP):
    if len(SEP) == 0:
        return []

    non_consecutive = [SEP[0]]

    for index in range(1, len(SEP)):
        if SEP[index][1] != SEP[index - 1][1] + 1:
            non_consecutive.append(SEP[index])

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


def compute_F1(precision: float, recall: float):
    if precision + recall > 0:
        return round((2 * precision * recall) / (precision + recall), 2)
    return 0.0


def compute_statistics(SEP, all_index_labels):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    FN, FP, TN, TP = count_statistics(FN, FP, SEP, TN, TP, all_index_labels)
    print_statistics(FN, FP, TN, TP)


# TODO: refactor this method to accept any CP_MATCH_INTERVAL
def count_statistics(FN, FP, SEP, TN, TP, all_index_labels):
    SEP_indexes = [tup[1] for tup in SEP]
    tps = []
    fps = []

    for sep in SEP_indexes:
        if CP_MATCH_INTERVAL == 0:
            if all_index_labels[sep - 1][1] != "Other_Activity" and all_index_labels[sep - 2][1] != "Other_Activity":
                if all_index_labels[sep - 1][1] != all_index_labels[sep - 2][1]:
                    TP = TP + 1
                    tps.append(sep)
                else:
                    FP = FP + 1
                    fps.append(sep)

        if CP_MATCH_INTERVAL == 1:
            if (all_index_labels[sep - 1][1] != "Other_Activity" and all_index_labels[sep - 2][
                1] != "Other_Activity") or \
                    (all_index_labels[sep][1] != "Other_Activity" and all_index_labels[sep - 1][
                        1] != "Other_Activity") or \
                    (all_index_labels[sep - 3][1] != "Other_Activity" and all_index_labels[sep - 2][
                        1] != "Other_Activity"):
                if all_index_labels[sep - 1][1] != all_index_labels[sep - 2][1] or \
                        all_index_labels[sep][1] != all_index_labels[sep - 1][1] or \
                        all_index_labels[sep - 3][1] != all_index_labels[sep - 2][1]:
                    TP = TP + 1
                    tps.append(sep)
                else:
                    FP = FP + 1
                    fps.append(sep)

        if CP_MATCH_INTERVAL == 2:
            if (all_index_labels[sep - 1][1] != "Other_Activity" and all_index_labels[sep - 2][
                1] != "Other_Activity") or \
                    (all_index_labels[sep][1] != "Other_Activity" and all_index_labels[sep - 1][
                        1] != "Other_Activity") or \
                    (all_index_labels[sep - 3][1] != "Other_Activity" and all_index_labels[sep - 2][
                        1] != "Other_Activity") or \
                    (all_index_labels[sep][1] != "Other_Activity" and all_index_labels[sep + 1][
                        1] != "Other_Activity") or \
                    (all_index_labels[sep - 3][1] != "Other_Activity" and all_index_labels[sep - 4][
                        1] != "Other_Activity"):
                if all_index_labels[sep - 1][1] != all_index_labels[sep - 2][1] or \
                        all_index_labels[sep][1] != all_index_labels[sep - 1][1] or \
                        all_index_labels[sep - 3][1] != all_index_labels[sep - 2][1] or \
                        all_index_labels[sep][1] != all_index_labels[sep + 1][1] or \
                        all_index_labels[sep - 3][1] != all_index_labels[sep - 4][1]:
                    TP = TP + 1
                    tps.append(sep)
                else:
                    FP = FP + 1
                    fps.append(sep)

    for i in range(1, len(all_index_labels)):
        current_index = all_index_labels[i][0]
        current_label = all_index_labels[i][1]

        previous_label = all_index_labels[i - 1][1]

        if current_index not in SEP_indexes and current_label != "Other_Activity" and previous_label != "Other_Activity":
            if current_label != previous_label:
                FN = FN + 1
            else:
                TN = TN + 1

    return FN, FP, TN, TP


def print_statistics(FN, FP, TN, TP):
    print("===============")
    print("TP: " + str(TP))
    print("TN: " + str(TN))
    print("FP: " + str(FP))
    print("FN: " + str(FN))
    precision = compute_precision(FP, TP)
    recall = compute_recall(FN, TP)
    print("==================")
    print("PRECISION: " + str(precision))
    print("RECALL: " + str(recall))
    print("ACCURACY: " + str(compute_accuracy(TP, TN, FP, FN)))
    print("F1 SCORE: " + str(compute_F1(precision, recall)))
    print("==================")


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
    arg_parser.add_argument('--threshold', type=float, required=True)
    arg_parser.add_argument('--CP_match_interval', type=int, required=True)

    # data parser
    arg = arg_parser.parse_args()
    THRESHOLD = arg.threshold
    CP_MATCH_INTERVAL = arg.CP_match_interval

    DEFAULT_SEP_LOCATION = 'C:/Users/FlaviusZichil/Desktop/MASTER/Anul 1/Semestrul 2/CERCETARE/Change-point-detection-AIMAS/pickles/'

    SEP = load_pickle('./pickles/HH103/time-normalized/HH103-w30-n3-k25.0-l0.1.pkl')
    all_index_labels = load_pickle('./pickles/index-labels/hh103-index-label.pkl')

    # this method is only used for sep files that used partial list of seps
    # SEP = extend_sep(SEP)

    print('Initial size: ' + str(len(SEP)))
    SEP = apply_threshold(SEP, THRESHOLD)
    print('After threshold: ' + str(len(SEP)))
    SEP = remove_consecutive_SEP_points(SEP)
    print('Without consecutive CPs: ' + str(len(SEP)))

    SEP_size = len(SEP)

    # this method computes precision, recall, accuracy and f1score
    # it takes into consideration a match interval (0, 1, 2) and ignores Other_Activity labels
    compute_statistics(SEP, all_index_labels)

    # this method counts how many sep points are really transitions in data set
    # in can take into consideration any CP_MATCH_INTERVAL > 0
    # count_number_of_sep_match(SEP, all_index_labels, CP_MATCH_INTERVAL)

    # computes report of matches between SEP point and actual transition from an activity and to that activity
    # results = compute_from_and_to_activity_match(all_index_labels, SEP, CP_MATCH_INTERVAL)
    # writeResultsInExcelFile(results, 'example.xls')
    # print(sum([result.accuracy for result in results]) / len(results))
