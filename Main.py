import timeit
from argparse import ArgumentParser

import numpy as np
from sklearn.gaussian_process.kernels import RBF

from Window import Window
from src.features.CountOfEventsFeature import CountOfEventsFeature
from src.features.DayOfWeekFeature import DayOfWeekFeature
from src.features.DominantLocationFeature import DominantLocationFeature
from src.features.EntropyFeature import EntropyFeature
from src.features.HourOfDayFeature import HourOfDayFeature
from src.features.LastSensorLocation import LastSensorLocationFeature
from src.features.MostFrequentSensorFeature import MostFrequentSensorFeature
from src.features.MostRecentSensorFeature import MostRecentSensorFeature
from src.features.NumberOfSensorEventsFeature import NumberOfSensorEventsFeature
from src.features.NumberOfTransitionsFeature import NumberOfTransitionsFeature
from src.features.SecondsPastMidNightFeature import SecondsPastMidNightFeature
from src.features.TimeBetweenEventsFeature import TimeBetweenEventsFeature
from src.features.WindowDurationFeature import WindowDurationFeature
from src.features.extractor.FeatureExtractor import FeatureExtractor
from src.utils.Encoder import Encoder
from src.utils.WindowEventsParser import WindowEventsParser


def build_features():
    number_of_sensor_events_feature = NumberOfSensorEventsFeature()
    window_duration_feature = WindowDurationFeature()
    most_recent_sensor_feature = MostRecentSensorFeature()
    most_frequent_sensor_feature = MostFrequentSensorFeature()
    last_sensor_location = LastSensorLocationFeature()
    dominant_location_feature = DominantLocationFeature()
    number_of_transitions_feature = NumberOfTransitionsFeature()
    count_of_events_feature = CountOfEventsFeature()
    absolute_time_between_events_feature = TimeBetweenEventsFeature('absolute')
    proportional_time_between_events_feature = TimeBetweenEventsFeature('proportional')
    entropy_feature = EntropyFeature()
    day_of_week_feature = DayOfWeekFeature()
    hour_of_day_feature = HourOfDayFeature()
    seconds_past_mid_night_feature = SecondsPastMidNightFeature()

    return [window_duration_feature,
            most_recent_sensor_feature,
            most_frequent_sensor_feature,
            last_sensor_location,
            dominant_location_feature,
            number_of_transitions_feature,
            count_of_events_feature,
            absolute_time_between_events_feature,
            proportional_time_between_events_feature,
            entropy_feature,
            hour_of_day_feature,
            day_of_week_feature,
            seconds_past_mid_night_feature]


def compute_H(N, current_x, previous_x):
    H = []

    for i in range(0, N):
        h = np.zeros((1, 2))
        for j in range(0, N):
            h = h + kernel.__call__(np.array(current_x[j]).reshape(1, len(current_x[j])),
                                    np.array(previous_x[i]).reshape(1, len(previous_x[i])))
        H.append(h)

    return H


def compute_theta(h, regularization_param):
    return -1 * np.array(h) / regularization_param


def compute_G(N, current_x, previous_x):
    G = []

    for i in range(0, N):
        g = np.zeros((1, 2))
        for j in range(0, N):
            g = g + theta[i] * kernel.__call__(np.array(previous_x[i]).reshape(1, len(previous_x[i])),
                                               np.array(current_x[j]).reshape(1, len(current_x[j])))
        G.append(g)

    return G


def compute_SEP(N, G):
    SEP = []

    for index in range(0, len(G) - N):
        SEP.append(max(0, np.sum(G[index:index + N])))

    return SEP


if __name__ == "__main__":
    start = timeit.default_timer()

    WINDOW_SIZE = 30

    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--file', type=str, required=True)

    # data parser
    arg = arg_parser.parse_args()
    data_set_file = arg.file
    parser = WindowEventsParser()
    parser.read_data_from_file(data_set_file)
    all_events = parser.events

    # features
    features = build_features()
    feature_extractor = FeatureExtractor(features)

    feature_windows = []
    encoded_feature_windows = []

    oneHotEncoder = Encoder()
    for i in range(0, len(all_events) - WINDOW_SIZE + 1):
        # get current 30 events window
        window = Window(all_events[i:WINDOW_SIZE + i])
        # print(count_of_events_feature.get_result(window))
        # get array of features from window
        feature_window = feature_extractor.extract_features_from_window(window)
        feature_windows.append(feature_window)
        # get one hot encoded array of features from previously created arrays of features
        # the index for feature_window comes from features array order
        encoded_feature_window = [feature_window[0]]
        encoded_feature_window.extend(oneHotEncoder.encode_attribute(feature_window[1], parser.sensor_names))
        encoded_feature_window.extend(oneHotEncoder.encode_attribute(feature_window[2], parser.sensor_names))
        encoded_feature_window.extend(oneHotEncoder.encode_attribute(feature_window[3], parser.sensor_locations))
        encoded_feature_window.extend(oneHotEncoder.encode_attribute(feature_window[4], parser.sensor_locations))
        encoded_feature_window.append(feature_window[5])
        encoded_feature_window.extend(feature_window[6])
        encoded_feature_window.extend(feature_window[7])
        encoded_feature_window.extend(feature_window[8])
        encoded_feature_window.append(feature_window[9])
        encoded_feature_window.append(feature_window[10])
        encoded_feature_window.append(feature_window[11])
        encoded_feature_window.append(feature_window[12])
        encoded_feature_windows.append(encoded_feature_window)

        # print(encoded_feature_window)

    G = []
    kernel = 1.0 * RBF(1.0)
    regularization_param = 1
    N = 3
    X = encoded_feature_windows

    for index in range(0, len(X) + 1 - 2 * N):
        previous_x = X[index:N + index]
        assert len(previous_x) == N

        current_x = X[N + index:2 * N + index]
        assert len(current_x) == N

        h = compute_H(N, current_x, previous_x)
        assert len(h) == N

        theta = compute_theta(h, regularization_param)
        assert len(theta) == N

        g = compute_G(N, current_x, previous_x)
        G.extend(g)

    SEP = compute_SEP(N, G)
    print(SEP)

    stop = timeit.default_timer()
    print('Time: ', stop - start)
