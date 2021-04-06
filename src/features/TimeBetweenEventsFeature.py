import datetime

from src.features.WindowDurationFeature import WindowDurationFeature
from src.features.base.Feature import Feature


class TimeBetweenEventsFeature(Feature):
    TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
    mode = ''

    def __init__(self, mode):
        self.name = 'Time between events ' + mode
        self.mode = mode

    def get_result(self, window):
        result = [0]
        events = window.events
        number_of_events = len(events)

        if self.mode == 'absolute':
            for i in range(1, number_of_events):
                result.append(self.get_time_between_two_events(events[i - 1], events[i]))

        if self.mode == 'proportional':
            window_duration = self.get_window_duration(window)

            for i in range(1, number_of_events):
                if window_duration != 0:
                    result.append(self.get_time_between_two_events(events[i - 1], events[i]) / window_duration)
                else:
                    result.append(0)

        return result

    def get_time_between_two_events(self, first_event, second_event):
        first_event_time = datetime.datetime.strptime(first_event.date + ' ' + first_event.time, self.TIME_FORMAT)
        second_event_time = datetime.datetime.strptime(second_event.date + ' ' + second_event.time, self.TIME_FORMAT)
        dt = second_event_time - first_event_time

        return dt.seconds / 60

    def get_window_duration(self, window):
        window_duration_feature = WindowDurationFeature()
        return window_duration_feature.get_result(window)