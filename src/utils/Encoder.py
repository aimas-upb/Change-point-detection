import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src.utils.WindowEventsParser import WindowEventsParser


class Encoder:
    def encode_attribute(self, attribute: str, all_values):
        sensor_names_encoder = OneHotEncoder(drop='first')

        encoded_values = sensor_names_encoder.fit_transform(all_values).toarray()
        decoded_values = sensor_names_encoder.inverse_transform(encoded_values)

        if len(all_values) > 0 and [attribute] in all_values:
            encoded_attribute_index = [x[0] for x in np.where(decoded_values == [attribute])][0]
            return encoded_values[encoded_attribute_index]