import time
from abc import abstractmethod
from datetime import datetime


class Feature:
    TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
    SEC = "seconds"
    MIN = "minutes"
    MOTION_SENSOR_PREFIX = "M"
    
    @abstractmethod
    def get_result(self, window):
        pass

    @staticmethod
    def get_event_timestamp(event):
        sensor_datetime = datetime.strptime(event.date + ' ' + event.time, Feature.TIME_FORMAT)
        sensor_ts = time.mktime(sensor_datetime.timetuple())
        
        return sensor_ts

    @staticmethod
    def get_datetime_diff(current_dt, previous_dt, metric=SEC):
        current_ts = time.mktime(current_dt.timetuple())
        previous_ts = time.mktime(previous_dt.timetuple())
    
        if metric == Feature.SEC:
            return current_ts - previous_ts
        else:
            return (current_ts - previous_ts) / 60

    @staticmethod
    def get_event_ts_diff(current_event, previous_event, metric=SEC):
        current_datetime = datetime.strptime(current_event.date + ' ' + current_event.time, Feature.TIME_FORMAT)
        previous_datetime = datetime.strptime(previous_event.date + ' ' + previous_event.time, Feature.TIME_FORMAT)
        
        return Feature.get_datetime_diff(current_datetime, previous_datetime, metric=metric)

    @staticmethod
    def is_motion_sensor(self, sensor_name: str):
        return sensor_name[0].startswith(self.MOTION_SENSOR_PREFIX)
