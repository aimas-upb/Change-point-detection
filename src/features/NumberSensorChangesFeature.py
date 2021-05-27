from src.features.base.Feature import Feature


class NumberSensorChangesFeature(Feature):

    def __init__(self, mode="proportional"):
        self.name = 'Number of Sensor Changes'
        self.mode = mode

    def get_result(self, window):
        number_of_transitions = 0
        events = window.events
        
        activation_events = [ev for ev in events if not Feature.is_motion_sensor(ev.sensor.name)
                             or ev.sensor.state == "ON"]

        for i in range(1, len(activation_events)):
            if activation_events[i].sensor.name != activation_events[i - 1].sensor.name:
                number_of_transitions = number_of_transitions + 1
    
        if not activation_events:
            return 0.0

        if self.mode == "absolute":
            return number_of_transitions
        return number_of_transitions / len(activation_events)