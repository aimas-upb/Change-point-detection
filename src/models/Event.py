class Event:
    def __init__(self, date, time, sensor, index, label):
        self.date = date
        self.time = time
        self.sensor = sensor
        self.index = index
        self.label = label

    def to_string(self):
        return "EVENT [date = " + self.date + ", time = " + self.time + ", " + self.sensor.to_string() + ", index = " + str(
            self.index) + ", label = " + (self.label) + "]"
