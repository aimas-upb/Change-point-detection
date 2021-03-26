from src.features.base.Feature import Feature
from src.utils.Encoder import Encoder
from src.utils.WindowEventsParser import WindowEventsParser


class LastSensorLocationFeature(Feature):
    def __init__(self):
        self.name = 'Last sensor location'

    # returns the location of the last activated sensor
    def get_result(self, window):
        oneHotEncoder = Encoder()
        parser = WindowEventsParser()
        return oneHotEncoder.encode_attribute(window.events[-1].sensor.location, parser.sensor_locations)
