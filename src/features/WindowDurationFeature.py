from src.features.base.Feature import Feature


class WindowDurationFeature(Feature):
    def __init__(self):
        self.name = 'Window duration'

    # returns the duration of the window
    def get_result(self, window):
        first_event = window.events[0]
        last_event = window.events[-1]

        return Feature.get_event_ts_diff(last_event, first_event, metric=Feature.MIN)

