import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
from sklearn.preprocessing import StandardScaler
import pandas as pd


def create_lstm_vae(input_dim,
                    timesteps,
                    batch_size,
                    intermediate_dim,
                    latent_dim,
                    epsilon_std=1.):
    """
    Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator.
    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM.
        latent_dim: int, latent z-layer shape.
        epsilon_std: float, z-layer sigma.
    """
    x = Input(shape=(timesteps, input_dim,))

    # LSTM encoding
    h = LSTM(intermediate_dim)(x)

    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

    print("PRIMA LAMBDA")
    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    print("DOPO LAMBDA")

    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    print("DOPO DECODED")

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)
    print("FINITO X_DECODED_MEAN")

    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    print("DOPO VAE")

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    print("DOPO LAMBDA")
    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    print("PRIMA DI GENERATOR")
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    print("COMPILO")
    vae.compile(optimizer='rmsprop', loss=vae_loss)
    print("FATTO")
    return vae, encoder, generator


def read_data(data_path):
    data = pd.read_csv(data_path)
    return data

boat_csv = read_data("Data/Boat_data.csv")
boat_csv = boat_csv.drop(columns=["Unnamed: 0"])

scaler = StandardScaler()
boat_csv = scaler.fit_transform(boat_csv)

print("RESHAPO")

boat_csv = boat_csv.values.reshape((boat_csv.shape[0], boat_csv.shape[1], 1))
print("FATTO")


vae, encoder, generator = create_lstm_vae(input_dim=boat_csv.shape[0],timesteps=10,batch_size=5,intermediate_dim=32,latent_dim=10)

print("MEGACIAO")
vae.fit(x=boat_csv,y=boat_csv, epochs=10)

print("MEGACIAO2")
