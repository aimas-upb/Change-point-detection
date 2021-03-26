from argparse import ArgumentParser

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
            hour_of_day_feature,
            day_of_week_feature,
            seconds_past_mid_night_feature,
            each_sensor_last_activation_time_feature]


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--file', type=str, required=True)
    arg_parser.add_argument('--window_length', type=int, required=True)
    arg_parser.add_argument('--event_index', type=int, required=True)

    # data parser
    arg = arg_parser.parse_args()
    data_set_file = arg.file
    WINDOW_LENGTH = arg.window_length
    EVENT_INDEX = arg.event_index

    parser = WindowEventsParser()
    parser.read_data_from_file(data_set_file)
    all_events = parser.events

    # features
    # defines the list of features that will be extracted from each window
    features = build_features()
    feature_extractor = FeatureExtractor(features)

    feature_windows = []
    oneHotEncoder = Encoder()

    last_window = Window(all_events[EVENT_INDEX - WINDOW_LENGTH - 1:EVENT_INDEX - 1])
    last_feature_window = feature_extractor.extract_features_from_window(last_window)
    print('------ LAST FEATURE WINDOW -------')
    # print(last_feature_window)
    print('Length: ' + str(len(last_feature_window)) + '\n')

    current_window = Window(all_events[EVENT_INDEX - WINDOW_LENGTH:EVENT_INDEX])
    current_feature_window = feature_extractor.extract_features_from_window(current_window)
    print('------ CURRENT FEATURE WINDOW -------')
    # print(current_feature_window)
    print('Length: ' + str(len(current_feature_window)) + '\n')

    next_window = Window(all_events[EVENT_INDEX - WINDOW_LENGTH + 1:EVENT_INDEX + 1])
    next_feature_window = feature_extractor.extract_features_from_window(next_window)
    print('------ NEXT FEATURE WINDOW -------')
    # print(next_feature_window)
    print('Length: ' + str(len(next_feature_window)))

    