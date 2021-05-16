class Window:
    def __init__(self, events):
        self.events = events

    def get_sensors(self):
        return [event.sensor for event in self.events]

    def get_sensor_names(self):
        window_sensors = self.get_sensors()
        return [sensor.name for sensor in window_sensors]

    def get_sensor_locations(self):
        window_sensors = self.get_sensors()
        return [sensor.location for sensor in window_sensors]

    @staticmethod
    def count_active_appearances(window, sensor_name):
        activations = 0

        for event in window.events:
            if event.sensor.name == sensor_name and event.sensor.state == "ON":
                activations = activations + 1

        return activations

    def to_string(self):
        print("WINDOW: ")
        for event in self.events:
            print(event.to_string())