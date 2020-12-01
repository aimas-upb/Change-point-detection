from src.features.base.Feature import Feature


class HourOfDayFeature(Feature):

    def __init__(self):
        self.name = 'Hour of day'

    def get_result(self, window):
        return int(window.events[0].time[0:2])