# -*- coding: utf-8 -*-

import optuna
import tensorflow as tf
from tensorflow import keras
from wgan_gp import WGAN, generator_loss, discriminator_loss, train_images, BATCH_SIZE, noise_dim

# number of epochs you want to train
epochs = 3

def objective(trial):
    g_model = keras.models.load_model('./optimized/Generator.h5')
    d_model = keras.models.load_model('./optimized/Discriminator.h5')

    #lr = trial.suggest_float("lr", 0.00001, 0.1, log=True)
    #b_1 = trial.suggest_float("beta_1", 0.5, 1.0, log=True)
    b_2 = trial.suggest_float("beta_2", 0.5, 1.0, log=True)

    generator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.0, beta_2=b_2, epsilon=1e-8
    )
    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.0, beta_2=b_2, epsilon=1e-8
    )

    newGanModel = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=noise_dim,
        discriminator_extra_steps=1,
    )

    # Compile the WGAN model.
    newGanModel.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )

    # Start training the model.
    history = newGanModel.fit(train_images, batch_size=BATCH_SIZE, epochs=epochs, verbose=0)
    loss = history.history['d_loss'][-1][-1][0][0][0]
    print("Actual loss: " + str(loss))
    if loss > 0:
        loss = 10
    return abs(loss)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))