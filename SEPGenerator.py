import os
import pickle
import timeit
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from densratio import densratio

from src.features.CountOfEventsFeature import CountOfEventsFeature
from src.features.DominantLocationFeature import DominantLocationFeature
from src.features.EachSensorLastActivationFeature import EachSensorLastActivationFeature
from src.features.EntropyFeature import EntropyFeature, TemporalEntropyFeature
from src.features.LastSensorLocation import LastSensorLocationFeature
from src.features.MostFrequentSensorFeature import MostFrequentSensorFeature
from src.features.MostRecentSensorFeature import MostRecentSensorFeature
from src.features.NumberOfTransitionsFeature import NumberOfTransitionsFeature
from src.features.TimeBetweenEventsFeature import TimeBetweenEventsFeature
from src.features.WindowDurationFeature import WindowDurationFeature
from src.features.extractor.FeatureExtractor import FeatureExtractor
from src.features.state.CountOfONEventsFeature import CountOfONEventsFeature
from src.features.state.EntropyONFeature import EntropyONFeature
from src.features.state.MostFrequentONSensorFeature import MostFrequentONSensorFeature
from src.features.state.MostRecentONSensorFeature import MostRecentONSensorFeature
from src.features.state.NumberOfONSensorEventsFeature import NumberOfONSensorEventsFeature
from src.features.state.TimeBetweenONEventsFeature import TimeBetweenONEventsFeature
from src.features.NumberSensorChangesFeature import NumberSensorChangesFeature
from src.models.Window import Window
from src.utils.Encoder import Encoder
from src.utils.WindowEventsParser import WindowEventsParser


def build_features_from_config(FEATURES):
    features = []

    for feature_class_name in FEATURES:
        if has_parameter(feature_class_name):
            feature_class_name = feature_class_name.replace('(\'', ' ').replace('\')', ' ')
            class_name = feature_class_name.split()[0]
            class_parameter = feature_class_name.split()[1]
            features.append(globals()[class_name](class_parameter))
        else:
            features.append(globals()[feature_class_name]())

    return features


def has_parameter(feature_class_name: str):
    return '(' in feature_class_name


def build_features():
    window_duration_feature = WindowDurationFeature()
    most_recent_sensor_feature = MostRecentSensorFeature()
    most_recent_ON_sensor_feature = MostRecentONSensorFeature()
    most_frequent_sensor_feature = MostFrequentSensorFeature()
    most_frequent_ON_sensor_feature = MostFrequentONSensorFeature()
    last_sensor_location = LastSensorLocationFeature()
    dominant_location_feature = DominantLocationFeature()
    number_of_transitions_feature = NumberOfTransitionsFeature()
    entropy_feature = EntropyFeature()
    entropy_ON_feature = EntropyONFeature()
    number_of_ON_sensor_events_feature = NumberOfONSensorEventsFeature()

    count_of_events_feature = CountOfEventsFeature()
    count_of_ON_events_feature = CountOfONEventsFeature()
    absolute_time_between_events_feature = TimeBetweenEventsFeature('absolute')
    absolute_time_between_ON_events_feature = TimeBetweenONEventsFeature('absolute')
    proportional_time_between_events_feature = TimeBetweenEventsFeature('proportional')
    proportional_time_between_ON_events_feature = TimeBetweenONEventsFeature('proportional')

    # day_of_week_feature = DayOfWeekFeature()
    # hour_of_day_feature = HourOfDayFeature()
    # seconds_past_mid_night_feature = SecondsPastMidNightFeature()
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
            # seconds_past_mid_night_feature,
            each_sensor_last_activation_time_feature]


def save_sep_data(file_name: str, SEP, SEP_assignments):
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
    
    if SEP_assignments:
        sep_assignment_file = file_name + "_sep_assign.txt"
        with open(sep_assignment_file, "w+") as f:
            for text in SEP_assignments:
                f.write(text)
                f.write("\n")


def add_statistics_to_dataset(sensor_index):
    current_window = Window(all_events[sensor_index - 1 - window_length:sensor_index - 1])
    previous_window = Window(all_events[sensor_index - 2 - window_length:sensor_index - 2])

    current_feature_window = feature_extractor.extract_features_from_window(current_window)
    previous_feature_window = feature_extractor.extract_features_from_window(previous_window)

    current_event = all_events[sensor_index - 1]

    event_details = str(current_event.index) + " " + \
                    str(current_event.date) + " " + \
                    str(current_event.time) + " " + \
                    str(current_event.sensor.name) + " " + \
                    str(current_event.sensor.state) + " " + \
                    str(current_event.sensor.location) + " " + \
                    str(current_event.label)
 
    spaces = ''

    for i in range(0, 55 - len(event_details)):
        spaces = spaces + ' '

    print(event_details + " " + spaces +
          " -> SEP: " + str((round(sep, 4))) + " " +
          " -> FEATURE WINDOW: " + str(
        [curr - prev for curr, prev in zip(current_feature_window, previous_feature_window)]))


def add_sep_assignment(sensor_index, sep, all_events, sep_assignments, feature_extractor, window_length):
    current_window = Window(all_events[sensor_index - 1 - window_length: sensor_index - 1])
    previous_window = Window(all_events[sensor_index - 2 - window_length: sensor_index - 2])
    
    current_feature_window = feature_extractor.extract_features_from_window(current_window)
    previous_feature_window = feature_extractor.extract_features_from_window(previous_window)
    
    current_event = all_events[sensor_index - 1]

    event_details = str(current_event.index) + " " + \
                    str(current_event.date) + " " + \
                    str(current_event.time) + " " + \
                    str(current_event.sensor.name) + " " + \
                    str(current_event.sensor.state) + " " + \
                    str(current_event.sensor.location) + " " + \
                    str(current_event.label)

    spaces = ''
    for i in range(0, 55 - len(event_details)):
        spaces = spaces + ' '
    
    event_details += spaces
    event_details += " -> SEP: " + str((round(sep, 4)))
    event_details += " -> FEATURE WINDOW: " + str(list(current_feature_window - previous_feature_window))
    
    sep_assignments.append(event_details)


if __name__ == "__main__":
    start = timeit.default_timer()

    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--file', type=str, required=True)
    arg_parser.add_argument('--window_length', type=int, required=True)
    arg_parser.add_argument('--N', type=int, required=True)

    # arguments matcher
    arg = arg_parser.parse_args()
    DATA_SET = arg.file
    window_length = arg.window_length
    N = arg.N

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
        for i in range(0, len(all_events) - window_length + 1):
            print(i)
            # get current 30 events window
            window = Window(all_events[i:window_length + i])
            # get array of features from window
            feature_window = feature_extractor.extract_features_from_window(window)
            feature_windows.append(feature_window)

        pickle.dump(feature_windows, open(dest_file, "wb"))

    # run RuLSIF experiments for different regularization and kernel param
    # kernel_param_grid = [1, 5, 10, 20]
    # regularization_param = [0.5, 0.75, 0.9]
    kernel_param_grid = [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 7.5, 10, 15, 20]
    regularization_param = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for sigma in kernel_param_grid:
        for lamda in regularization_param:

            res_file_name = dest_folder + source_file_name + "_res_%3.2f_%3.2f" % (sigma, lamda)

            SEP = []

            for index in range(N, len(feature_windows) + 1 - N):
                # print('Index SEP: ' + str(index) + '/' + str(len(feature_windows) + 1 - N))
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

                sensor_index = feature_windows.index(previous_x[N - 1]) + window_length

                add_statistics_to_dataset(sensor_index)

                SEP.append((round(sep, 4), sensor_index))

            save_sep_data(res_file_name, SEP)
            print(SEP)