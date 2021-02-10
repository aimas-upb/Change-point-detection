from src.features.base.Feature import Feature
from src.utils.Encoder import Encoder
from src.utils.WindowEventsParser import WindowEventsParser


class MostFrequentSensorFeature(Feature):
    def __init__(self):
        self.name = 'Most frequent sensor'

    # for previous_most_frequent_sensor call this for previous window
    def get_result(self, window):
        sensor_names = window.get_sensor_names()
        oneHotEncoder = Encoder()
        parser = WindowEventsParser()
        return oneHotEncoder.encode_attribute(max(set(sensor_names), key=sensor_names.count), parser.sensor_names)
