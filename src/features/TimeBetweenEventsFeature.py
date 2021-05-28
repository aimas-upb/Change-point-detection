from src.features.WindowDurationFeature import WindowDurationFeature
from src.features.base.Feature import Feature
import numpy as np


class TimeBetweenEventsFeature(Feature):
    mode = ''

    def __init__(self, mode):
        self.name = 'Time between events ' + mode
        self.mode = mode

    def get_result(self, window):
        result = []
        events = window.events
        number_of_events = len(events)

        for i in range(1, number_of_events):
            result.append(self.get_time_between_two_events(events[i], events[i - 1]))

        result = np.array(result)

        if self.mode == 'proportional':
            total_duration = np.sum(result)
            result = result / total_duration

        return result

    def get_time_between_two_events(self, first_event, second_event):
        return Feature.get_event_ts_diff(first_event, second_event, metric=Feature.MIN)


class AvgTimeBetweenActivations(Feature):
    def __init__(self, mode):
        self.name = 'Avg time between activations ' + mode
        self.mode = mode

    def get_result(self, window):
        result = []
        events = window.events
        number_of_events = len(events)

        prev_idx = 0
        while not Feature.is_activation(events[prev_idx]):
            prev_idx += 1

        crt_idx = prev_idx + 1
        while crt_idx < number_of_events:
            while not Feature.is_activation(events[crt_idx]) and crt_idx < number_of_events:
                crt_idx += 1

            if Feature.is_activation(events[crt_idx]):
                result.append(Feature.get_event_ts_diff(events[crt_idx], events[prev_idx], metric=Feature.MIN))
                prev_idx = crt_idx
                crt_idx = prev_idx + 1

        if not result:
            return 0.0

        result = np.array(result)

        if self.mode == 'proportional':
            total_duration = np.sum(result)
            result = result / total_duration

        return np.mean(result)


class MedianTimeBetweenActivations(Feature):
    def __init__(self, mode):
        self.name = 'Median time between activations ' + mode
        self.mode = mode

    def get_result(self, window):
        result = []
        events = window.events
        number_of_events = len(events)

        prev_idx = 0
        while not Feature.is_activation(events[prev_idx]):
            prev_idx += 1

        crt_idx = prev_idx + 1
        while crt_idx < number_of_events:
            while not Feature.is_activation(events[crt_idx]) and crt_idx < number_of_events:
                crt_idx += 1

            if Feature.is_activation(events[crt_idx]):
                result.append(Feature.get_event_ts_diff(events[crt_idx], events[prev_idx], metric=Feature.MIN))
                prev_idx = crt_idx
                crt_idx = prev_idx + 1

        if not result:
            return 0.0

        result = np.array(result)

        if self.mode == 'proportional':
            total_duration = np.sum(result)
            result = result / total_duration

        return np.median(result)
