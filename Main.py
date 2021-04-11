import os
import pickle
import timeit
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from densratio import densratio

from Window import Window
from src.features.CountOfEventsFeature import CountOfEventsFeature
from src.features.DayOfWeekFeature import DayOfWeekFeature
from src.features.DominantLocationFeature import DominantLocationFeature
from src.features.EachSensorLastActivationFeature import EachSensorLastActivationFeature
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
    entropy_feature = EntropyFeature()

    count_of_events_feature = CountOfEventsFeature()
    absolute_time_between_events_feature = TimeBetweenEventsFeature('absolute')
    proportional_time_between_events_feature = TimeBetweenEventsFeature('proportional')

    day_of_week_feature = DayOfWeekFeature()
    hour_of_day_feature = HourOfDayFeature()
    seconds_past_mid_night_feature = SecondsPastMidNightFeature()
    each_sensor_last_activation_time_feature = EachSensorLastActivationFeature()

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
            seconds_past_mid_night_feature,
            each_sensor_last_activation_time_feature]

    # return [each_sensor_last_activation_time_feature]


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


def compute_H2(A):
    return A.sum(axis=1) / N


def build_A(N, current_x, previous_x):
    A = []

    for i in range(0, N):
        a = []
        for j in range(0, N):
            kernel_value = kernel.__call__(np.array(current_x[j]).reshape(1, len(current_x[j])),
                                     np.array(previous_x[i]).reshape(1, len(previous_x[i])))
            a.append(kernel_value[0][0])
        A.append(a)

    result = np.array(A)

    assert (N, N) == result.shape
    return result


def functie(h, lamda):
    h = np.array(h).reshape((len(h), 1))
    print(h.shape)
    # return lambda theta: theta.T @ (h * h @ theta + lamda * theta @ theta
    return lambda theta: theta.T @ (h @ h.T) @ theta + lamda * theta.T @ theta


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


def display_sep_distribution():
    sep_values = [tup[0] for tup in SEP]
    s = pd.Series(sep_values)
    print(s.describe())
    sns.boxplot(data=sep_values)
    plt.show()
    
    
def save_sep_data(file_name: str, SEP):
    sep_df = pd.DataFrame(SEP, columns=["sep", "sensor_index"])
    
    summary_file = file_name + "_summary.txt"
    with open(summary_file, "w+") as f:
        f.write(str(sep_df["sep"].describe()))
        f.write("\n\n")
        f.write("Num non-zero SEPs: %i" % sum(sep_df["sep"] > 0))
        f.write("\n\n")
        f.write("non-zero SEPs: %s" % str(sep_df.loc[sep_df["sep"] > 0]))
        
    plot_file = file_name + "_plot.svg"
    fig = plt.figure()
    res_boxplot = sep_df.boxplot(column=["sep"])
    plt.savefig(plot_file, format="svg")

    pkl_file = file_name + ".pkl"
    pickle.dump(SEP, open(pkl_file, "wb"))


if __name__ == "__main__":
    start = timeit.default_timer()

    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--file', type=str, required=True)
    arg_parser.add_argument('--window_length', type=int, required=True)
    arg_parser.add_argument('--N', type=int, required=True)
    # arg_parser.add_argument('--kernel_param', type=float, required=True)
    # arg_parser.add_argument('--l', type=float, required=True)

    # arguments matcher
    arg = arg_parser.parse_args()
    DATA_SET = arg.file
    WINDOW_LENGTH = arg.window_length
    N = arg.N
    # KERNEL_PARAM = arg.kernel_param
    # REGULARIZATION_PARAM = arg.l

    # data parser
    parser = WindowEventsParser()
    parser.read_data_from_file(DATA_SET)
    all_events = parser.events
    
    # features
    # defines the list of features that will be extracted from each window
    features = build_features()
    feature_extractor = FeatureExtractor(features)

    feature_windows = []
    oneHotEncoder = Encoder()
    
    source_file_name = os.path.splitext(os.path.basename(DATA_SET))[0]
    dest_folder = "src" + os.path.sep + "results" + os.path.sep
    dest_file = dest_folder + source_file_name + ".pkl"
    
    if os.path.exists(dest_file):
        feature_windows = pickle.load(open(dest_file, "rb"))
    else:
        for i in range(0, len(all_events) - WINDOW_LENGTH + 1):
            print(i)
            # get current 30 events window
            window = Window(all_events[i:WINDOW_LENGTH + i])
            # get array of features from window
            feature_window = feature_extractor.extract_features_from_window(window)
            feature_windows.append(feature_window)

        pickle.dump(feature_windows, open(dest_file, "wb"))
        
    # run RuLSIF experiments for different regularization and kernel param
    kernel_param = [1, 5, 10, 20]
    regularization_param = [0.5, 0.75, 0.9]
    
    for sigma in kernel_param:
        for lamda in regularization_param:
            
            res_file_name = dest_folder + source_file_name + "_res_%3.2f_%3.2f" % (sigma, lamda)
    
            SEP = []

            for index in range(N, len(feature_windows) + 1 - N):
                print('Index SEP: ' + str(index) + '/' + str(len(feature_windows) + 1 - N))
                previous_x = feature_windows[index - N:index]
                assert len(previous_x) == N
        
                current_x = feature_windows[index:N + index]
                assert len(current_x) == N
        
                # use previous_x as the Y samples for distribution of f_(t-1)(x) and
                # use current_x as the X samples for distribution f_t(x) in a call to densratio RuLSIF -
                densratio_res = densratio(x=np.array(current_x), y=np.array(previous_x), kernel_num=len(previous_x),
                                          sigma_range=[sigma], lambda_range=[lamda],
                                          verbose=False)
                
                g_sum = np.sum(densratio_res.compute_density_ratio(np.array(current_x))) / len(current_x)
                sep = max(0, 0.5 - g_sum)
        
                sensor_index = feature_windows.index(previous_x[N - 1]) + WINDOW_LENGTH
                SEP.append((round(sep, 4), sensor_index))
                
            save_sep_data(res_file_name, SEP)