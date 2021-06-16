import pickle
from argparse import ArgumentParser
from collections import Counter

import numpy as np
import yaml
from sklearn.ensemble import GradientBoostingClassifier

from SEPGenerator import build_features_from_config
from src.features.extractor.FeatureExtractor import FeatureExtractor
from src.models.Window import Window
from src.utils.WindowEventsParser import WindowEventsParser


def load_configurations(config_file_path: str):
    with open(config_file_path) as config_file:
        return yaml.load(config_file)


def load_change_points(config_file_path: str):
    return pickle.load(open(config_file_path, "rb"))


def distinct_days(block):
    event_dates = [event.date for event in block]
    return set(event_dates)


def get_excluded_days(ALL_EVENTS, fold):
    excluded_days = []

    all_days = list(distinct_days(ALL_EVENTS))
    all_days.sort()

    for index in range(0, fold):
        excluded_days.append(all_days[index])

    # print("EXLCUDED DAYS FOR FOLD " + str(fold) + " ARE " + str(excluded_days))
    return excluded_days


def build_blocks(ALL_EVENTS, fold):
    blocks = []
    block = []

    excluded_days = get_excluded_days(ALL_EVENTS, fold)

    for event in ALL_EVENTS:
        if event.date in excluded_days:
            continue

        distinct = distinct_days(block)

        if len(block) > 0:
            last_added_event = block[len(block) - 1]

        if len(distinct) < 6 or (len(distinct) == 6 and event.date == last_added_event.date):
            block.append(event)

        if len(distinct) == 6 and event.date != last_added_event.date:
            blocks.append(block)
            block = []

    if len(distinct_days(block)) == 6:
        blocks.append(block)

    return blocks


def get_train_and_test_data(ALL_EVENTS, fold):
    train_data = []
    test_data = []

    blocks = build_blocks(ALL_EVENTS, fold)

    for block in blocks:
        distinct = list(distinct_days(block))
        distinct.sort()

        train_dates = distinct[0:4]
        test_dates = distinct[4:6]

        # print("TRAIN DATES: " + str(train_dates))
        # print("TEST DATES: " + str(test_dates))

        for event in block:
            if event.date in train_dates:
                train_data.append(event)
            if event.date in test_dates:
                test_data.append(event)

    return train_data, test_data


def get_dominant_label(window):
    labels = [event.label for event in window.events]
    return max(set(labels), key=labels.count)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--config', type=str, required=True)
    arg = arg_parser.parse_args()

    CONFIGURATIONS = load_configurations(arg.config)

    CHANGE_POINTS_FILE = CONFIGURATIONS['change-points']
    DATA_SET = CONFIGURATIONS['data-set']
    FEATURES = CONFIGURATIONS['features']
    FOLDS = CONFIGURATIONS['folds']

    change_points = [tup[1] for tup in load_change_points(CHANGE_POINTS_FILE)]

    parser = WindowEventsParser()
    parser.read_data_from_file(DATA_SET)
    ALL_EVENTS = parser.events

    features = build_features_from_config(FEATURES)
    feature_extractor = FeatureExtractor(features)

    scores = []

    for fold in range(0, FOLDS):
        TRAIN_DATA, TEST_DATA = get_train_and_test_data(ALL_EVENTS, fold)

        print(len(TRAIN_DATA))
        print(len(TEST_DATA))

        min_train_index = TRAIN_DATA[0].index
        max_train_index = TRAIN_DATA[len(TRAIN_DATA) - 1].index
        X_TRAIN = []
        Y_TRAIN = []

        min_test_index = TEST_DATA[0].index
        max_test_index = TEST_DATA[len(TEST_DATA) - 1].index
        X_TEST = []
        Y_TEST = []
        fw_length = 0

        for change_point_index in range(1, len(change_points)):
            activity_start_index = change_points[change_point_index - 1]
            activity_end_index = change_points[change_point_index]

            if min_train_index <= activity_start_index <= max_train_index and min_train_index <= activity_end_index <= max_train_index:
                window = Window(ALL_EVENTS[activity_start_index:activity_end_index])

                feature_window = feature_extractor.extract_features_from_window(window)
                fw_length = len(feature_window)
                # print(len(feature_window))
                X_TRAIN.extend(feature_window)
                Y_TRAIN.append(get_dominant_label(window))

            if min_test_index <= activity_start_index <= max_test_index and min_test_index <= activity_end_index <= max_test_index:
                window = Window(ALL_EVENTS[activity_start_index:activity_end_index])

                feature_window = feature_extractor.extract_features_from_window(window)
                X_TEST.extend(feature_window)
                Y_TEST.append(get_dominant_label(window))

        print(str(Counter(Y_TRAIN)))
        print(str(Counter(Y_TEST)))

        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)
        # clf = RandomForestClassifier(random_state=0, class_weight="balanced")

        X_TRAIN = np.reshape(X_TRAIN, (len(Y_TRAIN), fw_length))
        X_TEST = np.reshape(X_TEST, (len(Y_TEST), fw_length))

        clf.fit(X_TRAIN, Y_TRAIN)
        score = clf.score(X_TEST, Y_TEST)

        scores.append(score)

    print(scores)

    average_score = np.average(scores)
    std = np.std(scores)

    print('AVG: ' + str(average_score))
    print('STD: ' + str(std))

