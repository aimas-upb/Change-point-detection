from src.features.base.Feature import Feature
from src.utils.Encoder import Encoder
from src.utils.WindowEventsParser import WindowEventsParser


class MostRecentSensorFeature(Feature):
    def __init__(self):
        self.name = 'Most recent sensor'

    # returns last sensor of window
    def get_result(self, window):
        oneHotEncoder = Encoder()
        parser = WindowEventsParser()
        return oneHotEncoder.encode_attribute(window.events[-1].sensor.name, parser.sensor_names)

