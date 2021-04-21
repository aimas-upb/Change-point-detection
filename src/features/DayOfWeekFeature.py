import datetime

from src.features.base.Feature import Feature

DATE_FORMAT = '%Y-%m-%d'


class DayOfWeekFeature(Feature):

    def __init__(self):
        self.name = 'Day of week'

    # TODO may not be taken into consideration for SEP points
    def get_result(self, window):
        return datetime.datetime.strptime(window.events[-1].date, DATE_FORMAT).weekday() / 7
    