from src.features.WindowDurationFeature import WindowDurationFeature
from src.features.base.Feature import Feature


class TimeBetweenONEventsFeature(Feature):
    mode = ''

    def __init__(self, mode):
        self.name = 'Time between ON events ' + mode
        self.mode = mode

    def get_result(self, window):
        result = [0]
        events = window.events
        number_of_events = len(events)

        latest_active_event = events[0]

        for i in range(1, number_of_events):
            if (Feature.is_motion_sensor(events[i].sensor.name) and events[i].sensor.state == "ON") or \
                    not Feature.is_motion_sensor(events[i].sensor.name):
                result.append(self.get_time_between_two_events(events[i], latest_active_event))
                latest_active_event = events[i]
            else:
                result.append(0)

        if self.mode == 'absolute':
            return result

        if self.mode == 'proportional':
            window_duration = self.get_window_duration(window)
            return [res / window_duration for res in result]

    def get_time_between_two_events(self, first_event, second_event):
        return Feature.get_event_ts_diff(first_event, second_event, metric=Feature.MIN)

    def get_window_duration(self, window):
        window_duration_feature = WindowDurationFeature()
        return window_duration_feature.get_result(window)
