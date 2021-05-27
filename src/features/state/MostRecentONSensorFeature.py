from src.features.base.Feature import Feature
from src.utils.Encoder import Encoder
from src.utils.WindowEventsParser import WindowEventsParser


class MostRecentONSensorFeature(Feature):
    def __init__(self):
        self.name = 'Most recent ON sensor'

    # returns last sensor of window
    def get_result(self, window):
        oneHotEncoder = Encoder()
        parser = WindowEventsParser()

        for event in reversed(window.events):
            if Feature.is_motion_sensor(event.sensor.name):
                if event.sensor.state == "ON":
                    return oneHotEncoder.encode_attribute(event.sensor.name, parser.sensor_names)
            else:
                return oneHotEncoder.encode_attribute(event.sensor.name, parser.sensor_names)
