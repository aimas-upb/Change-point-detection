from src.features.base.Feature import Feature
from src.models.Window import Window

from src.utils.WindowEventsParser import WindowEventsParser


class CountOfONEventsFeature(Feature):

    def __init__(self):
        self.name = 'Count of ON  motion events feature'

    def get_result(self, window):
        result = []
        parser = WindowEventsParser()
        all_sensor_names = parser.sensor_names
        sensors_names_from_window = [event.sensor.name for event in window.events]

        for sensor_name in all_sensor_names:
            if Feature.is_motion_sensor(sensor_name):
                result.append(Window.count_active_appearances(window, sensor_name) / len(window.events))
            else:
                result.append(sensors_names_from_window.count(sensor_name) / len(window.events))

        return result
