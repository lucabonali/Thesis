import keras
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


class Data_loader:
    def __init__(self):
        pass

    def read_data(self, path):
        data = pd.read_csv(path)
        return data

    def load_data(self, path, index_to_remove):
        csv = self.read_data(path)
        csv = csv.drop(columns=["Unnamed: 0"])
        csv = csv.drop(csv.index[-index_to_remove:])
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
