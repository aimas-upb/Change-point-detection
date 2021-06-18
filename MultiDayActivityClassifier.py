import pickle
from argparse import ArgumentParser
import pprint

import numpy as np
import pandas as pd
import yaml
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from SEPGenerator import build_features_from_config
from src.features.extractor.FeatureExtractor import FeatureExtractor
from src.models.Window import Window
from src.utils.WindowEventsParser import WindowEventsParser
from datetime import datetime, timedelta


def load_configurations(config_file_path: str):
    with open(config_file_path) as config_file:
        return yaml.load(config_file)


def load_change_points(config_file_path: str):
    return pickle.load(open(config_file_path, "rb"))


def     distinct_days(block):
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


def get_all_labels(ALL_EVENTS):
    labels = set([])
    for ev in ALL_EVENTS:
        if not ev.label in labels:
            labels.add(ev.label)
    
    return np.array(list(labels))


def build_blocks(ALL_EVENTS, fold):
    blocks = []
    
    start_date = datetime.strptime(ALL_EVENTS[0].date, "%Y-%m-%d")
    end_date = datetime.strptime(ALL_EVENTS[-1].date, "%Y-%m-%d")
    
    crt_start = start_date + timedelta(days=fold)
    event_idx = 0

    while datetime.strptime(ALL_EVENTS[event_idx].date, "%Y-%m-%d") < crt_start:
        event_idx += 1
        
    while crt_start < end_date and event_idx < len(ALL_EVENTS):
        crt_end = crt_start + timedelta(days=5)
        block = []
        unique_dates = set([])
        while event_idx < len(ALL_EVENTS) and datetime.strptime(ALL_EVENTS[event_idx].date, "%Y-%m-%d") < crt_end:
            block.append(ALL_EVENTS[event_idx])
            ev_date = datetime.strptime(ALL_EVENTS[event_idx].date, "%Y-%m-%d")
            if not ev_date in unique_dates:
                unique_dates.add(ev_date)
            
            event_idx += 1
        
        if len(unique_dates) == 6:
            blocks.append(block)
            
        crt_start = crt_end + timedelta(days=1)
        
    return blocks
    
    # excluded_days = get_excluded_days(ALL_EVENTS, fold)

    # for event in ALL_EVENTS:
    #     if event.date in excluded_days:
    #         continue
    #
    #     distinct = distinct_days(block)
    #
    #     if len(block) > 0:
    #         last_added_event = block[len(block) - 1]
    #
    #     if len(distinct) < 6 or (len(distinct) == 6 and event.date == last_added_event.date):
    #         block.append(event)
    #
    #     if len(distinct) == 6 and event.date != last_added_event.date:
    #         blocks.append(block)
    #         block = []
    #
    # if len(distinct_days(block)) == 6:
    #     blocks.append(block)
    #
    # return blocks


def get_train_and_test_blocks(ALL_EVENTS, fold):

    blocks = build_blocks(ALL_EVENTS, fold)
    
    train_blocks = []
    test_blocks = []
    
    for block in blocks:
        distinct = list(distinct_days(block))
        distinct.sort()

        train_dates = distinct[0:4]
        test_dates = distinct[4:6]

        # print("TRAIN DATES: " + str(train_dates))
        # print("TEST DATES: " + str(test_dates))
        train_block = []
        test_block = []
        
        for event in block:
            if event.date in train_dates:
                train_block.append(event)
            if event.date in test_dates:
                test_block.append(event)
        
        print(distinct)
        
        train_blocks.append(train_block)
        test_blocks.append(test_block)
        
    return train_blocks, test_blocks


def get_dominant_label(window):
    labels = [event.label for event in window.events]
    return max(set(labels), key=labels.count)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--config', type=str, required=True)
    arg = arg_parser.parse_args()
    pp = pprint.PrettyPrinter(indent=4)
    
    
    CONFIGURATIONS = load_configurations(arg.config)

    CHANGE_POINTS_FILE = CONFIGURATIONS['change-points']
    DATA_SET = CONFIGURATIONS['data-set']
    FEATURES = CONFIGURATIONS['features']
    FOLDS = CONFIGURATIONS['folds']

    change_points = [tup[1] for tup in load_change_points(CHANGE_POINTS_FILE)]

    parser = WindowEventsParser()
    parser.read_data_from_file(DATA_SET)
    ALL_EVENTS = parser.events
    ALL_EVENT_LABELS = np.array([ev.label for ev in ALL_EVENTS])
    
    label_df = pd.DataFrame(np.asarray((np.unique(ALL_EVENT_LABELS, return_counts=True))).T, columns=["label", "freq"])
    label_df["freq"] = label_df["freq"].astype(int)
    label_df["freq"] = label_df["freq"] / label_df["freq"].max()
    FREQ_LABELS = label_df[label_df["freq"] >= 1e-2]["label"].to_list()
    
    features = build_features_from_config(FEATURES)
    feature_extractor = FeatureExtractor(features)

    fold_scores = []
    
    for fold in range(0, FOLDS):
        TRAIN_BLOCKS, TEST_BLOCKS = get_train_and_test_blocks(ALL_EVENTS, fold)
        
        confusion_mat = None
        
        for block_idx in range(len(TRAIN_BLOCKS)):
            train_data = TRAIN_BLOCKS[block_idx]
            test_data = TEST_BLOCKS[block_idx]
            
            min_train_index = train_data[0].index
            max_train_index = train_data[len(train_data) - 1].index
            X_TRAIN = []
            Y_TRAIN = []
    
            min_test_index = test_data[0].index
            max_test_index = test_data[len(test_data) - 1].index
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

            # print(str(Counter(Y_TRAIN)))
            # print(str(Counter(Y_TEST)))

            # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)
            # clf = RandomForestClassifier(random_state=0, class_weight="balanced")

            clf = RandomForestClassifier(n_estimators=200, criterion='gini', class_weight="balanced")
            
            X_TRAIN = np.reshape(X_TRAIN, (len(Y_TRAIN), fw_length))
            X_TEST = np.reshape(X_TEST, (len(Y_TEST), fw_length))
    
            clf.fit(X_TRAIN, Y_TRAIN)
            Y_PRED = clf.predict(X_TEST)
            
            # y_test_weights = compute_sample_weight(class_weight="balanced", y=Y_TEST)
            
            if confusion_mat is None:
                confusion_mat = confusion_matrix(Y_TEST, Y_PRED, labels=FREQ_LABELS)
            else:
                res = confusion_matrix(Y_TEST, Y_PRED, labels=FREQ_LABELS)
                confusion_mat = confusion_mat + res

        fold_score_dict = {}
        
        fold_acc = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat)
        fold_score_dict["acc"] = fold_acc

        for label_idx in range(len(FREQ_LABELS)):
            label = FREQ_LABELS[label_idx]
    
            label_prec = 0
            label_recall = 0
            label_f1 = 0
    
            if np.sum(confusion_mat[:, label_idx]) > 0:
                label_prec = confusion_mat[label_idx, label_idx] / np.sum(confusion_mat[:, label_idx])
    
            if np.sum(confusion_mat[label_idx]) > 0:
                label_recall = confusion_mat[label_idx, label_idx] / np.sum(confusion_mat[label_idx])
    
            if label_prec > 0 or label_recall > 0:
                label_f1 = 2 * label_prec * label_recall / (label_prec + label_recall)
    
            fold_score_dict[label] = {
                "f1": label_f1,
                "prec": label_prec,
                "rec": label_recall
            }
        
        
        print("#### [INFO] FOLD score dict ...")
        pp.pprint(fold_score_dict)
        print("\n")
        fold_scores.append(fold_score_dict)
        
    
    cv_scores = {}
    accuracies = [d["acc"] for d in fold_scores]
    
    cv_scores["acc"] = [d["acc"] for d in fold_scores]
    for i in range(len(FREQ_LABELS)):
        label = FREQ_LABELS[i]
        cv_scores[label + "_" + "f1"] = [d[label]["f1"] for d in fold_scores]
        cv_scores[label + "_" + "precision"] = [d[label]["prec"] for d in fold_scores]
        cv_scores[label + "_" + "recall"] = [d[label]["rec"] for d in fold_scores]
    
    cv_score_df = pd.DataFrame.from_dict(cv_scores)
    print(cv_score_df.describe())
    
    src_filename = os.path.splitext(os.path.basename(CHANGE_POINTS_FILE))[0]
    cv_score_file = src_filename + "_" + "cv_scores.pkl"
    cv_score_filepath = os.path.dirname(CHANGE_POINTS_FILE) + os.path.sep + cv_score_file
    
    cv_score_df.to_pickle(cv_score_filepath)
    
    # print("General AVG acc: " + str(np.average(accuracies)))
    # print("General AVG std: " + str(np.std(accuracies)))
    #
    # for i in range(len(FREQ_LABELS)):
    #     label = FREQ_LABELS[i]
    #     label_avg_f1 = [d[label]["f1"] for d in fold_scores]
    #     label_avg_prec = [d[label]["prec"] for d in fold_scores]
    #     label_avg_rec = [d[label]["rec"] for d in fold_scores]
    #
    #     print(label + " AVG f1: " + str(np.average(label_avg_f1)))
    #     print(label + " STD f1: " + str(np.std(label_avg_f1)))
    #
    #     print(label + " AVG prec: " + str(np.average(label_avg_prec)))
    #     print(label + " STD prec: " + str(np.std(label_avg_prec)))
    #
    #     print(label + " AVG rec: " + str(np.average(label_avg_rec)))
    #     print(label + " STD rec: " + str(np.std(label_avg_rec)))
