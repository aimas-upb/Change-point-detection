from src.features.base.Feature import Feature


class NumberOfTransitionsFeature(Feature):

    def __init__(self):
        self.name = 'Number of transitions'

    def get_result(self, window):
        number_of_transitions = 0
        events = window.events

        for i in range(1, len(events)):
            if events[i].sensor.location != events[i - 1].sensor.location:
                number_of_transitions = number_of_transitions + 1

        return number_of_transitions
