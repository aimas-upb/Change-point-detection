import codecs
import pickle
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from src.utils.WindowEventsParser import WindowEventsParser


def get_interval_length(tuple):
    return tuple[1] - tuple[0]


def safe_open(input_file_path, rw="r"):
    try:
        input_file = codecs.open(input_file_path, rw, 'utf-8')
    except UnicodeDecodeError:
        input_file = codecs.open(input_file_path, rw, 'utf-16')
    return input_file


def get_activity_intervals(activity, events):
    intervals = []
    index = 1
    start = 1

    while index < len(events):
        if events[index - 1].label == activity:
            if events[index].label != events[index - 1].label:
                if start != index:
                    intervals.append((start, index))
                start = index + 1
        else:
            start = start + 1
        index = index + 1

    return intervals


def get_longest_activity_instances(activity_intervals, n):
    activity_intervals.sort(key=get_interval_length, reverse=True)
    longest_activity_intervals = activity_intervals[0:n]

    for interval in longest_activity_intervals:
        print(str(interval) + ' -> ' + str(get_interval_length(interval)))

    return longest_activity_intervals


def get_sep_points(sep_file):
    SEP_points = []

    with open(sep_file, 'rb') as file:
        for sep in pickle.load(file):
            SEP_points.append(sep)

    return SEP_points


def is_index_in_interval(sep_index, activity_interval):
    return sep_index >= activity_interval[0] and sep_index <= activity_interval[1]


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


def get_sensor_name_for_event_index(events, sep_index):
    for event in events:
        if event.index == sep_index:
            return event.sensor.name


def print_most_frequent_sensors_for_each_activity(most_frequent_sensor_names):
    print('Most frequent wrong CP sensor')
    for interval, sensor_name in most_frequent_sensor_names:
        print(str(interval) + ' -> ' + str(sensor_name))
    print('------------------')


def print_wrong_SEP_raport_for_each_activity(wrong_SEP_raport):
    print('Wrong SEP points raport')
    for interval, raport in wrong_SEP_raport:
        print(str(interval) + ' -> ' + str(raport))
    print('------------------')


def plot_results(all_points_to_plot, lenghts):
    fig, axs = plt.subplots(len(all_points_to_plot))
    fig.suptitle('Activity segmentation (HH' + arg.HH + ', Activity: ' + ACTIVITY + ', Threshold: ' + str(THRESHOLD) + ')')

    for index in range(0, len(all_points_to_plot)):
        points = [point[0] for point in all_points_to_plot[index]]
        labels = [point[1] for point in all_points_to_plot[index]]

        axs[index].plot(points, np.ones(len(points)), marker='o')
        axs[index].legend([lenghts[index]], loc='lower right')
        i = 0

        for x, y in zip(points, np.ones(len(points))):

            axs[index].annotate(labels[i],  # this is the text
                             (x, y),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(-5, 10),  # distance from text to points (x,y)
                             ha='left',
                             rotation=75)
            i = i + 1
    plt.show()


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--HH', type=str, required=True)
    arg_parser.add_argument('--SEP_file', type=str, required=True)
    arg_parser.add_argument('--activity', type=str, required=True)
    arg_parser.add_argument('--n', type=int, required=True)
    arg_parser.add_argument('--T', type=float, required=True)

    arg = arg_parser.parse_args()
    DATA_SET = arg.HH
    SEP_FILE = arg.SEP_file
    ACTIVITY = arg.activity
    # top n activities of type ACTIVITY
    N = arg.n
    THRESHOLD = arg.T

    parser = WindowEventsParser()
    parser.read_data_from_file(DATA_SET)
    events = parser.events

    activity_intervals = get_activity_intervals(ACTIVITY, events)
    longest_activity_intervals = get_longest_activity_instances(activity_intervals, N)
    SEP_points = get_sep_points(SEP_FILE)
    print("Initial number of SEP points: " + str(len(SEP_points)))
    SEP_points = apply_threshold(SEP_points, THRESHOLD)
    print("Number of SEP points after threshold: " + str(len(SEP_points)))
    SEP_points = remove_consecutive_SEP_points(SEP_points)
    print("Number of SEP points after removing consecutive points: " + str(len(SEP_points)))
    SEP_indexes = [tup[1] for tup in SEP_points]

    most_frequent_sensor_names = []
    wrong_SEP_raport = []
    all_points_to_plot = []
    lengths = []

    if len(SEP_indexes) > 0:
        for activity_interval in longest_activity_intervals:
            print(activity_interval[0])
            current_activity_points_to_plot = [(activity_interval[0], get_sensor_name_for_event_index(events, activity_interval[0]))]
            sensor_names = []
            wrong_SEP_counts = 0

            for sep_label, sep_index in SEP_points:
                if is_index_in_interval(sep_index, activity_interval):
                    print('a')
                    sensor_name = get_sensor_name_for_event_index(events, sep_index)
                    sensor_names.append(sensor_name)
                    wrong_SEP_counts = wrong_SEP_counts + 1
                    current_activity_points_to_plot.append((sep_index, sensor_name))
                    print('--' + str(sep_index) + ' -- ' + sensor_name + ' -- ' + str(sep_label))

            if len(sensor_names) > 0:
                most_frequent_sensor_names.append((activity_interval, max(set(sensor_names), key=sensor_names.count)))

            wrong_SEP_raport.append((activity_interval, round(wrong_SEP_counts / get_interval_length(activity_interval), 2)))
            lengths.append(str(wrong_SEP_counts) + '/' + str(get_interval_length(activity_interval)) + ' (wrong/total)')

            current_activity_points_to_plot.append((activity_interval[0], get_sensor_name_for_event_index(events, activity_interval[0])))
            all_points_to_plot.append(current_activity_points_to_plot)
            print(activity_interval[1])
            print('------------------')

    print_most_frequent_sensors_for_each_activity(most_frequent_sensor_names)
    print_wrong_SEP_raport_for_each_activity(wrong_SEP_raport)

    print("Total number of activities: " + str(len(activity_intervals)))

    print(all_points_to_plot)
    print(lengths)
    plot_results(all_points_to_plot, lengths)