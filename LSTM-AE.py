
from keras.layers import *
from keras.models import Model as Model
from keras.layers import Input, LSTM, RepeatVector
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def read_data(data_path):
    data = pd.read_csv(data_path)
    return data

boat_csv = read_data("Data/Boat_data.csv")
boat_csv = boat_csv.drop(columns=["Unnamed: 0"])

scaler = StandardScaler()
data  = scaler.fit_transform(boat_csv)

#print(data)

trainX = np.reshape(data, (1, data.shape[0], data.shape[1])) # Batch size (Number of samples), time steps and number of features

timesteps=data.shape[0]
input_dim=7
batch_size=1
epochs=20

def get_model(n_dimensions):
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(n_dimensions, return_sequences=False, name="encoder")(inputs)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True, name='decoder')(decoded)

    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    return autoencoder, encoder

print("GETTING the model")
autoencoder, encoder = get_model(data.shape[1])
print("COMPILING")
autoencoder.compile(optimizer='adam', loss='mse',
                    metrics=['acc', 'cosine_proximity'])
print("FITTING")
history = autoencoder.fit(trainX, trainX, batch_size=100, epochs=20)
print("PREDICTING")
encoded = encoder.predict(trainX)

plt.plot()
print("End")