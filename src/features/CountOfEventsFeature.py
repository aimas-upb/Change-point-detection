from src.features.base.Feature import Feature


class CountOfEventsFeature(Feature):

    def __init__(self):
        self.name = 'Count of events'

    def get_result(self, window):
        sensors_names = [event.sensor.name for event in window.events]
        unique_sensor_names = set(sensors_names)
        return [round(sensors_names.count(sensor_name) / len(unique_sensor_names), 3) for sensor_name in unique_sensor_names]
