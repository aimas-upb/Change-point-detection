from src.features.WindowDurationFeature import WindowDurationFeature
from src.features.base.Feature import Feature
import datetime


class TimeBetweenEventsFeature(Feature):
    TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
    mode = ''

    def __init__(self, mode):
        self.name = 'Count of events'
        self.mode = mode

    def get_result(self, window):
        result = [0]
        events = window.events

        if self.mode == 'absolute':
            for i in range(1, len(events)):
                result.append(self.get_time_between_two_events(events[i], events[i - 1]))

        if self.mode == 'proportional':
            for i in range(1, len(events)):
                result.append(self.get_time_between_two_events(events[i], events[i - 1]) / self.get_window_duration(window))

        return result

    def get_time_between_two_events(self, first_event, second_event):
        first_event_time = datetime.datetime.strptime(first_event.date + ' ' + first_event.time, self.TIME_FORMAT)
        second_event_time = datetime.datetime.strptime(second_event.date + ' ' + second_event.time, self.TIME_FORMAT)
        return (second_event_time - first_event_time).microseconds / 1e6

    def get_window_duration(self, window):
        window_duration_feature = WindowDurationFeature()
        return window_duration_feature.get_result(window)