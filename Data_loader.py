import keras
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


class DataLoader:
    def __init__(self, path, index_to_remove):
        self.path = path
        self.index_to_remove = index_to_remove

    def read_data(self):
        data = pd.read_csv(self.path)
        return data

    def load_data(self):
        csv = self.read_data()
        csv = csv.drop(columns=["Unnamed: 0"])
        csv = csv.drop(csv.index[:-self.index_to_remove])
        scaler = StandardScaler()
        data = scaler.fit_transform(csv)
        return data

    def prepare_sequences(self, data, timesteps, interval):
        samples = []
        for i in range(0, data.shape[0] - timesteps, interval):
            sample = data[i:i + timesteps]
            samples.append(sample)

        sequences = np.array(samples)

        # Batch size (Number of samples time steps and number of features
        trainX = np.reshape(sequences, (len(samples), timesteps, 7))

        return trainX

    def load_model(self, model_path):
        return keras.models.load_model(model_path)
