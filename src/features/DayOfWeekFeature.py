from src.features.base.Feature import Feature
import datetime

DATE_FORMAT = '%Y-%m-%d'


class DayOfWeekFeature(Feature):

    def __init__(self):
        self.name = 'Day of week'

    def get_result(self, window):
        return datetime.datetime.strptime(window.events[0].date, DATE_FORMAT).weekday() / 7