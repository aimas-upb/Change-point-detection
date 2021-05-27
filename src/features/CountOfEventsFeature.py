from src.features.base.Feature import Feature
from src.utils.WindowEventsParser import WindowEventsParser

class CountOfEventsFeature(Feature):

    def __init__(self, mode="proportional"):
        self.name = 'Count of events feature'
        self.mode = mode

    def get_result(self, window):
        result = []
        parser = WindowEventsParser()
        all_sensor_names = parser.sensor_names
        sensors_names_from_window = [event.sensor.name for event in window.events]

        for sensor_name in all_sensor_names:
            sensor_ct = sensors_names_from_window.count(sensor_name)
            if self.mode == "proportional":
                result.append(sensor_ct / len(window.events))
            else:
                result.append(sensor_ct)

        return result
