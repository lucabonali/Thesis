from keras import Sequential
from keras.layers import LSTM, Dense, RepeatVector


class LSTM_AE_Model():

    def __init__(self, hidden_dimensions,
                 intermediate_dimension,
                 timesteps_sequences,
                 input_dim,
                 optimizer,
                 loss,
                 metrics):

        encoder = Sequential(name="encode")
        encoder.add(LSTM(timesteps_sequences, return_sequences=True))
        encoder.add(LSTM(intermediate_dimension, return_sequences=False))
        encoder.add(Dense(hidden_dimensions))

        decoder = Sequential(name="decode")
        decoder.add(Dense(hidden_dimensions))
        decoder.add(Dense(intermediate_dimension))
        decoder.add(RepeatVector(timesteps_sequences))
        decoder.add(LSTM(input_dim, return_sequences=True))

        autoencoder = Sequential()
        autoencoder.add(encoder)
        autoencoder.add(decoder)

        autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = autoencoder


    def fit_model(self,x, y, batch_size, epochs):
        history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs)
        return history

    def save_model(self, path):
        self.model.save("Models/", path)

    def set_model(self, model):
        self.model = model

    def encode(self, input):
        encoding = self.model.get_layer("encode").predict(input)
        return encoding

    def autoencode(self, input):
        autoencoded = self.model.predict(input)
        return autoencoded
