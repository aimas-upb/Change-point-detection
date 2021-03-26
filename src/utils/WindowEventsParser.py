import codecs

import numpy as np

from Event import Event
from Sensor import Sensor


class WindowEventsParser:
    events = []
    sensor_names = []
    sensor_locations = []

    def read_data_from_file(self, file):
        data_file = self.safe_open(file)
        lines = data_file.readlines()
        labels = []

        for index in range(0, len(lines)):
            elements = lines[index].split()
            date = elements[0]
            time = elements[1]
            label = elements[-1]

            # checks if data has locations
            if len(elements) == 4:
                sensor = Sensor(elements[2], elements[3], None)

                if elements[2] not in self.sensor_names:
                    self.sensor_names.append([elements[2]])
            else:
                sensor = Sensor(elements[3], elements[4], elements[2])

                if elements[3] not in self.sensor_names:
                    self.sensor_names.append(np.array([elements[3]]))

                if elements[2] not in self.sensor_locations:
                    self.sensor_locations.append(np.array([elements[2]]))

            self.events.append(Event(date, time, sensor))

            labels.append((index + 1, label))

    def safe_open(self, input_file_path, rw="r"):
        try:
            input_file = codecs.open(input_file_path, rw, 'utf-8')
        except UnicodeDecodeError:
            input_file = codecs.open(input_file_path, rw, 'utf-16')
        return input_file
