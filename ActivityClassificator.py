import pickle
from argparse import ArgumentParser

import yaml

from SEPGenerator import build_features_from_config
from src.features.extractor.FeatureExtractor import FeatureExtractor
from src.models.Window import Window
from src.utils.WindowEventsParser import WindowEventsParser


def load_configurations(config_file_path: str):
    with open(config_file_path) as config_file:
        return yaml.load(config_file)


def load_change_points(config_file_path: str):
    return pickle.load(open(config_file_path, "rb"))


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--config', type=str, required=True)
    arg = arg_parser.parse_args()

    CONFIGURATIONS = load_configurations(arg.config)

    CHANGE_POINTS_FILE = CONFIGURATIONS['change-points']
    DATA_SET = CONFIGURATIONS['data-set']
    TRAIN_DATA = CONFIGURATIONS['train-data']
    TEST_DATA = CONFIGURATIONS['test-data']
    FEATURES = CONFIGURATIONS['features']

    parser = WindowEventsParser()
    parser.read_data_from_file(DATA_SET)
    all_events = parser.events

    # change_points = load_change_points(CHANGE_POINTS_FILE)
    change_points = [100, 170, 247, 356, 444, 576, 600, 650, 720, 867, 900]

    features = build_features_from_config(FEATURES)
    feature_extractor = FeatureExtractor(features)

    feature_windows = []

    for change_point_index in range(1, len(change_points)):
        activity_start_index = change_points[change_point_index - 1]
        activity_end_index = change_points[change_point_index]

        window = Window(all_events[activity_start_index:activity_end_index])

        feature_window = feature_extractor.extract_features_from_window(window)
        feature_windows.append(feature_window)

    print(feature_windows)
    print(len(feature_windows))
