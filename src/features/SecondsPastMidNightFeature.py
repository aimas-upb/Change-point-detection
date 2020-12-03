from src.features.base.Feature import Feature
from datetime import datetime


class SecondsPastMidNightFeature(Feature):

    def __init__(self):
        self.name = 'Seconds past mid night'

    def get_result(self, window):
        midnight_time = datetime.now().replace(year=1900, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        current_date = datetime.strptime(window.events[0].time, '%H:%M:%S.%f')
        return (current_date - midnight_time).total_seconds() / 86400