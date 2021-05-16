from src.features.base.Feature import Feature


class NumberOfONSensorEventsFeature(Feature):
    def __init__(self):
        self.name = 'Number of ON sensor events'

    # returns the number of the window events
    def get_result(self, window):
        ON_sensor_events_count = 0

        for event in window.events:
            if self.is_motion_sensor(self, event.sensor.name):
                if event.sensor.state == "ON":
                    ON_sensor_events_count = ON_sensor_events_count + 1
            else:
                ON_sensor_events_count = ON_sensor_events_count + 1

        return ON_sensor_events_count
