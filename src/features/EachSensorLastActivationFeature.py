from datetime import datetime

from src.features.base.Feature import Feature
from src.models.Window import Window
from src.utils.WindowEventsParser import WindowEventsParser


class EachSensorLastActivationFeature(Feature):
    TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

    main_time_tracker = {}

    def __init__(self):
        self.name = 'Last activation time of each sensor'
        self.initialize_main_time_tracker()

    def get_result(self, window):
        parser = WindowEventsParser()
        all_sensor_names = parser.sensor_names

        result = [0] * len(all_sensor_names)

        for index in range(0, len(all_sensor_names)):
            current_sensor_name = all_sensor_names[index][0]

            current_sensor_time = self.get_last_sensor_activation_from_window(current_sensor_name, window)
            previous_sensor_time = self.main_time_tracker[current_sensor_name]
            
            if current_sensor_time:
                diff = Feature.get_datetime_diff(current_sensor_time, previous_sensor_time, metric=Feature.MIN)
                result[index] = diff
                self.main_time_tracker[current_sensor_name] = current_sensor_time
            else:
                current_sensor_time = datetime.strptime(window.events[-1].date + ' ' + window.events[-1].time,
                                                        self.TIME_FORMAT)
                diff = Feature.get_datetime_diff(current_sensor_time, previous_sensor_time, metric=Feature.MIN)
                result[index] = diff
            
        return result

    def get_last_sensor_activation_from_window(self, sensor: str, window: Window):
        for event in reversed(window.events):
            if event.sensor.name == sensor:
                return datetime.strptime(event.date + ' ' + event.time, self.TIME_FORMAT)

        return None

    def initialize_main_time_tracker(self):
        parser = WindowEventsParser()
        all_sensor_names = parser.sensor_names
        first_event = parser.events[0]
        first_event_time = datetime.strptime(first_event.date + ' ' + first_event.time, self.TIME_FORMAT)

        if not self.main_time_tracker:
            for sensor in all_sensor_names:
                self.main_time_tracker[sensor[0]] = first_event_time

        assert len(all_sensor_names) == len(self.main_time_tracker)
