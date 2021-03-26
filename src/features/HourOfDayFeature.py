from src.features.base.Feature import Feature


class HourOfDayFeature(Feature):

    def __init__(self):
        self.name = 'Hour of day'

    # TODO may not be taken into consideration for SEP points
    def get_result(self, window):
        return int(window.events[-1].time[0:2]) / 24