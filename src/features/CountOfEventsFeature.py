from src.features.base.Feature import Feature
from src.utils.WindowEventsParser import WindowEventsParser


class CountOfEventsFeature(Feature):

    def __init__(self):
        self.name = 'Count of events feature'

    def get_result(self, window):
        result = []
        parser = WindowEventsParser()
        all_sensor_names = parser.sensor_names
        sensors_names_from_window = [event.sensor.name for event in window.events]

        for sensor_name in all_sensor_names:
            result.append(round(sensors_names_from_window.count(sensor_name) / len(window.events), 2))

        return result
