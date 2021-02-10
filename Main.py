import timeit
from argparse import ArgumentParser

import numpy as np
import scipy.optimize as optimize
from sklearn.gaussian_process.kernels import RBF
import pickle


from Window import Window
from src.features.CountOfEventsFeature import CountOfEventsFeature
from src.features.DayOfWeekFeature import DayOfWeekFeature
from src.features.DominantLocationFeature import DominantLocationFeature
from src.features.EntropyFeature import EntropyFeature
from src.features.HourOfDayFeature import HourOfDayFeature
from src.features.LastSensorLocation import LastSensorLocationFeature
from src.features.MostFrequentSensorFeature import MostFrequentSensorFeature
from src.features.MostRecentSensorFeature import MostRecentSensorFeature
from src.features.NumberOfTransitionsFeature import NumberOfTransitionsFeature
from src.features.SecondsPastMidNightFeature import SecondsPastMidNightFeature
from src.features.TimeBetweenEventsFeature import TimeBetweenEventsFeature
from src.features.WindowDurationFeature import WindowDurationFeature
from src.features.extractor.FeatureExtractor import FeatureExtractor
from src.utils.Encoder import Encoder
from src.utils.WindowEventsParser import WindowEventsParser


def build_features():
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
        h = np.zeros((1, 1))
        for j in range(0, N):
            kernel_val = kernel.__call__(np.array(current_x[j]).reshape(1, len(current_x[j])),
                                    np.array(previous_x[i]).reshape(1, len(previous_x[i])))

            h = h + kernel_val
        H.append(h[0][0] / N)

    return H


def functie(h, lamda):
   h = np.array(h)
   return lambda theta: theta @ h * h @ theta + lamda * theta @ theta


def compute_G(N, current_x, previous_x, theta):
    G = []

    for i in range(0, N):
        g = np.zeros((1, 2))
        for j in range(0, N):
            g = g + theta[i] * kernel.__call__(np.array(previous_x[i]).reshape(1, len(previous_x[i])),
                                               np.array(current_x[j]).reshape(1, len(current_x[j])))
        G.append(g[0][0])

    return G


def compute_SEP(N, G):
    return max(0, 0.5 - (np.sum(G)) / N)


if __name__ == "__main__":
    start = timeit.default_timer()

    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--file', type=str, required=True)
    arg_parser.add_argument('--window_length', type=int, required=True)
    arg_parser.add_argument('--N', type=int, required=True)
    arg_parser.add_argument('--kernel_param', type=float, required=True)

    # data parser
    arg = arg_parser.parse_args()
    data_set_file = arg.file
    WINDOW_LENGTH = arg.window_length
    N = arg.N
    KERNEL_PARAM = arg.kernel_param

    parser = WindowEventsParser()
    parser.read_data_from_file(data_set_file)
    all_events = parser.events

    # features
    # defines the list of features that will be extracted from each window
    features = build_features()
    feature_extractor = FeatureExtractor(features)

    feature_windows = []
    oneHotEncoder = Encoder()

    for i in range(0, len(all_events) - WINDOW_LENGTH + 1):
        # get current 30 events window
        window = Window(all_events[i:WINDOW_LENGTH + i])
        # get array of features from window
        feature_window = feature_extractor.extract_features_from_window(window)
        feature_windows.append(feature_window)

    SEP = []
    partial_SEP = []
    kernel = 1.0 * RBF(KERNEL_PARAM)
    regularization_param = 1

    for index in range(N, len(feature_windows) + 1 - N):
        print('Index SEP: ' + str(index) + '/' + str(len(feature_windows) + 1 - N))
        previous_x = feature_windows[index - N:index]
        assert len(previous_x) == N

        current_x = feature_windows[index:N + index]
        assert len(current_x) == N

        h = compute_H(N, current_x, previous_x)
        assert len(h) == N

        opt_res = optimize.minimize(functie(h, regularization_param), np.ones((len(h),)), constraints=optimize.NonlinearConstraint(lambda x: sum(abs(x) - 1e-1), 0, np.inf))
        theta = opt_res.x
        assert len(theta) == N

        g = compute_G(N, current_x, previous_x, theta)

        sensor_index = feature_windows.index(previous_x[N - 1]) + WINDOW_LENGTH

        sep = compute_SEP(N, g)
        partial_SEP.append((sep, sensor_index))
        # used a partial list to avoid keeping the main list too large
        if len(partial_SEP) == 5000:
            SEP.append(partial_SEP)
            partial_SEP = []

    SEP.append(partial_SEP)

    print(SEP)

    # with open('pickles/HH101/bune/hh-test.pkl', 'wb') as file:
    #     pickle.dump(SEP, file)

    stop = timeit.default_timer()
    print('Time: ', stop - start)
