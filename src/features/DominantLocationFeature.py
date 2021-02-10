from src.features.base.Feature import Feature
from src.utils.Encoder import Encoder
from src.utils.WindowEventsParser import WindowEventsParser


class DominantLocationFeature(Feature):
    def __init__(self):
        self.name = 'Dominant location'

    # returns the location with the most sensor activations
    def get_result(self, window):
        sensor_locations = window.get_sensor_locations()
        oneHotEncoder = Encoder()
        parser = WindowEventsParser()
        return oneHotEncoder.encode_attribute(max(set(sensor_locations), key=sensor_locations.count), parser.sensor_locations)
