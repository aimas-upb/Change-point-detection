from src.features.base.Feature import Feature
from src.utils.Encoder import Encoder
from src.utils.WindowEventsParser import WindowEventsParser


class MostFrequentONSensorFeature(Feature):
    def __init__(self):
        self.name = 'Most frequent ON sensor'

    # for previous_most_frequent_sensor call this for previous window
    def get_result(self, window):
        sensor_names = self.filter_active_sensors_events(window)
        oneHotEncoder = Encoder()
        parser = WindowEventsParser()
        return oneHotEncoder.encode_attribute(max(sensor_names, key=sensor_names.count), parser.sensor_names)

    def filter_active_sensors_events(self, window):
        active_events_sensor_names = []

        for event in window.events:
            if self.is_motion_sensor(self, event.sensor.name):
                if event.sensor.state == "ON":
                    active_events_sensor_names.append(event.sensor.name)
            else:
                active_events_sensor_names.append(event.sensor.name)

        return active_events_sensor_names
