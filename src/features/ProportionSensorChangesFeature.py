from src.features.base.Feature import Feature


class ProportionSensorChangesFeature(Feature):

    def __init__(self):
        self.name = 'Number of Sensor Changes'

    def get_result(self, window):
        number_of_transitions = 0
        events = window.events
        
        activation_events = [ev for ev in events if not Feature.is_motion_sensor(self, ev.sensor.name)
                             or ev.sensor.state == "ON"]

        for i in range(1, len(activation_events)):
            if activation_events[i].sensor.name != activation_events[i - 1].sensor.name:
                number_of_transitions = number_of_transitions + 1
    
        if not activation_events:
            return 0.0
        
        return number_of_transitions / len(activation_events)
