# -*- coding: utf-8 -*-
"""wgan_gp

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/wgan_gp.ipynb

# WGAN-GP overriding `Model.train_step`

**Author:** [A_K_Nain](https://twitter.com/A_K_Nain)<br>
**Date created:** 2020/05/9<br>
**Last modified:** 2020/05/9<br>
**Description:** Implementation of Wasserstein GAN with Gradient Penalty.

## Wasserstein GAN (WGAN) with Gradient Penalty (GP)

The original [Wasserstein GAN](https://arxiv.org/abs/1701.07875) leverages the
Wasserstein distance to produce a value function that has better theoretical
properties than the value function used in the original GAN paper. WGAN requires
that the discriminator (aka the critic) lie within the space of 1-Lipschitz
functions. The authors proposed the idea of weight clipping to achieve this
constraint. Though weight clipping works, it can be a problematic way to enforce
1-Lipschitz constraint and can cause undesirable behavior, e.g. a very deep WGAN
discriminator (critic) often fails to converge.

The [WGAN-GP](https://arxiv.org/pdf/1704.00028.pdf) method proposes an
alternative to weight clipping to ensure smooth training. Instead of clipping
the weights, the authors proposed a "gradient penalty" by adding a loss term
that keeps the L2 norm of the discriminator gradients close to 1.

## Setup
"""
import math
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


"""## Prepare the Fashion-MNIST data

To demonstrate how to train WGAN-GP, we will be using the
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Each
sample in this dataset is a 28x28 grayscale image associated with a label from
10 classes (e.g. trouser, pullover, sneaker, etc.)
"""

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

img_shape_config = [
    (4, 4, 1),
    (8, 8, 1),
    (16, 16, 1),
    (32, 32, 1),
    (64, 64, 1),
    (128, 128, 1),
    (256, 256, 1)
]

IMG_SHAPE = (4, 4, 1)
EPOCHS_PER_SIZE = 4
TOTAL_EPOCHS = EPOCHS_PER_SIZE * ((len(img_shape_config)-1) * 2 + 1)
BATCH_SIZE = 16
FILTER_DEPTH = 4
CURRENT_SIZE = 0
CURRENT_TRANSITION = tf.Variable(0.0)
TRANSITION_SPEED = 0.0
IS_TRANSITION = False


# Size of the noise vector
noise_dim = 512

train_images = []
# 86176
for index in range(1024):
    image = cv2.imread("data\\out\\" + str(index) + ".png", cv2.IMREAD_GRAYSCALE)
    train_images.append(image)
    train_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    train_images.append(cv2.rotate(image, cv2.ROTATE_180))
    train_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    image = cv2.flip(image, 1)
    train_images.append(image)
    train_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    train_images.append(cv2.rotate(image, cv2.ROTATE_180))
    train_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))

train_images = np.array(train_images, dtype="float32")
train_images = train_images.reshape(train_images.shape[0], *(256, 256, 1))
print(train_images.shape)
train_images = (train_images - 127.5) / 127.5
np.random.seed(3213)
np.random.shuffle(train_images)


def pixel_norm(x, epsilon=1e-8):
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)


def lerp(a, b, t):
    return a + (b - a) * t


def conv_block(x, depth, strides, kernel_size, depth_multiplier=2, weights=None):
    l_1 = layers.Conv2D(FILTER_DEPTH * depth, kernel_size=max(3, kernel_size - 1), strides=1, padding="same", use_bias=True)
    x = l_1(x)
    if weights:
        l_1.set_weights(weights[0])
    x = layers.LeakyReLU(0.2)(x)
    l_2 = layers.Conv2D(FILTER_DEPTH * depth * depth_multiplier, kernel_size=kernel_size, strides=1, padding="same",
                      use_bias=True)
    x = l_2(x)
    if weights:
        l_2.set_weights(weights[1])
    x = layers.LeakyReLU(0.2)(x)
    x = layers.AveragePooling2D(strides)(x)
    return x


def d_transition_block(x, depth, strides, kernel_size, depth_multiplier=2, weights=None):
    y = layers.AveragePooling2D(strides)(x)
    y = layers.Conv2D(FILTER_DEPTH * depth * depth_multiplier, kernel_size=1, strides=1, padding="same", use_bias=True)(y)
    y = layers.LeakyReLU(0.2)(y)

    x = layers.Conv2D(FILTER_DEPTH * depth, kernel_size=1, strides=1, padding="same", use_bias=True)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = conv_block(x, depth, strides, kernel_size, depth_multiplier, weights)

    x = lerp(y, x, CURRENT_TRANSITION)
    return x


d_layer_config = [
    [128, 4, 5, 1],
    [128, 2, 3, 1],
    [64, 2, 3, 2],
    [32, 2, 3, 2],
    [16, 2, 3, 2],
    [8, 2, 3, 2],
    [4, 2, 3, 2],
]


def get_discriminator_model():
    img_input = layers.Input(shape=IMG_SHAPE)

    x = layers.Conv2D(FILTER_DEPTH * 128, kernel_size=1, strides=1, padding="same", use_bias=True)(img_input)
    x = layers.LeakyReLU(0.2)(x)

    x = conv_block(x, 128, 4, 5, 1)

    x = layers.Dense(1)(x)

    d_model = keras.models.Model(img_input, x, name="discriminator")
    return d_model


def deconv_block(x, depth, strides, kernel_size):
    x = layers.UpSampling2D(strides)(x)
    x = layers.Conv2D(filters=FILTER_DEPTH * depth, kernel_size=kernel_size, strides=1, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = pixel_norm(x)
    x = layers.Conv2D(filters=FILTER_DEPTH * depth, kernel_size=max(3, kernel_size - 1), strides=1, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = pixel_norm(x)
    return x


def g_transition_block(x, depth, strides, kernel_size):
    y = layers.UpSampling2D(strides)(x)
    x = deconv_block(x, depth, strides, kernel_size)
    x = layers.Conv2D(filters=1, kernel_size=1, strides=1, padding="same")(x)
    x = layers.Activation("linear")(x)

    y = layers.Conv2D(filters=1, kernel_size=1, strides=1, padding="same")(y)
    y = layers.Activation("linear")(y)

    x = lerp(y, x, CURRENT_TRANSITION)
    return x


g_layer_config = [
    [128, 4, 5],
    [128, 2, 3],
    [64, 2, 3],
    [32, 2, 3],
    [16, 2, 3],
    [8, 2, 3],
    [4, 2, 3]
]


def get_generator_model():
    noise = layers.Input(shape=(noise_dim,))
    x = layers.Reshape((1, 1, FILTER_DEPTH * 128))(noise)

    x = deconv_block(x, 128, 4, 5)
    x = layers.Conv2D(filters=1, kernel_size=1, strides=1, padding="same")(x)
    x = layers.Activation("linear")(x)
    """x = deconv_block(x, 128, 2, 3)
    x = deconv_block(x, 64, 2, 3)
    x = deconv_block(x, 32, 2, 3)
    x = deconv_block(x, 16, 2, 3)
    x = deconv_block(x, 8, 2, 3)"""

    #x = g_transition_block(x, 4, 2, 3) #256

    g_model = keras.models.Model(noise, x, name="generator")
    return g_model


if __name__ == "__main__":
    d_model = get_discriminator_model()
    d_model.summary()
    g_model = get_generator_model()
    g_model.summary()

"""## Create the WGAN-GP model

Now that we have defined our generator and discriminator, it's time to implement
the WGAN-GP model. We will also override the `train_step` for training.
"""


class WGAN(keras.Model):
    def __init__(
            self,
            discriminator,
            generator,
            latent_dim,
            discriminator_extra_steps=3,
            gp_weight=10,  # Penalty Coefficicent
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        if IS_TRANSITION:
            small_images = tf.image.resize(real_images, (int(IMG_SHAPE[0]/2), int(IMG_SHAPE[1]/2)))
            small_images = tf.image.resize(small_images, IMG_SHAPE[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if IMG_SHAPE[0] != img_shape_config[-1][0]:
            real_images = tf.image.resize(real_images, IMG_SHAPE[:2])

        """out = (tf.gather(small_img, 0) * 127.5) + 127.5
        out = tf.cast(out, tf.uint8)
        out = tf.image.encode_png(out)
        tf.io.write_file("test_s.png", out)"""
        if IS_TRANSITION:
            real_images = lerp(small_images, real_images, CURRENT_TRANSITION)

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight
                epsilon_penalty = tf.square(real_logits)
                d_loss += epsilon_penalty * 0.001

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


"""## Create a Keras callback that periodically saves generated images"""


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim), seed=3342934)

    def on_train_batch_end(self, step, logs=None):
        if IS_TRANSITION:
            CURRENT_TRANSITION.assign_add(TRANSITION_SPEED)
        """if step % 128 != 0:
            return

        generated_images = self.model.generator(self.random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5
        #CURRENT_TRANSITION.assign_add(128 / 512)
        print(CURRENT_TRANSITION)
        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.preprocessing.image.array_to_img(img)
            img.save("output\\{epoch}_heightmap_{i}.png".format(i=i, epoch=step))"""

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            return
        if epoch % EPOCHS_PER_SIZE == 0:
            current_step = int(epoch / EPOCHS_PER_SIZE)
            current_network = int(math.ceil(current_step / 2))
            print("Changing networks on stage " + str(current_network))
            global IS_TRANSITION
            if current_step % 2 == 0:
                print("Changing to non-transition")
                IS_TRANSITION = False
                self.transition_discriminator(current_network)
                self.transition_generator(current_network)
            else:
                print("Changing to transition")
                IS_TRANSITION = True
                global IMG_SHAPE
                IMG_SHAPE = img_shape_config[current_network]
                CURRENT_TRANSITION.assign(0.0)
                self.extend_discriminator(current_network)
                self.extend_generator(current_network)


    def on_epoch_end(self, epoch, logs=None):
        """
            TODO: Sequencing
        """
        generated_images = self.model.generator(self.random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.preprocessing.image.array_to_img(img)
            img.save("output\\{epoch}_heightmap_{i}.png".format(i=i, epoch=epoch))

        # Saving network only possible when not in resolution transition phase
        if not IS_TRANSITION:
            self.model.generator.save("models\\Generator_{epoch}.h5".format(epoch=epoch))
            self.model.discriminator.save("models\\Discriminator.h5")

    def extend_generator(self, config):
        config = g_layer_config[config]
        self.model.generator = keras.models.Model(self.model.generator.input, g_transition_block(self.model.generator.layers[-3].output, config[0], config[1], config[2]), name="new_generator")
        self.model.generator.summary()

    def transition_generator(self, config):
        self.model.generator = keras.models.Model(self.model.generator.input, self.model.generator.layers[-4].output, name="new_generator")
        self.model.generator.summary()

    def extend_discriminator(self, config):
        config = d_layer_config[config]
        d_layers = self.model.discriminator.layers[3:]
        print(IMG_SHAPE)
        input_layer = layers.Input(shape=IMG_SHAPE)
        x = d_transition_block(input_layer, config[0], config[1], config[2], config[3])
        for layer in d_layers:
            x = layer(x)
        self.model.discriminator = keras.models.Model(input_layer, x, name="new_discriminator")
        self.model.discriminator.summary()

    def transition_discriminator(self, config):
        config = d_layer_config[config]
        d_layers = self.model.discriminator.layers[14:]
        weights = []
        weights.append(self.model.discriminator.layers[1].get_weights())
        weights.append(self.model.discriminator.layers[3].get_weights())
        weights.append(self.model.discriminator.layers[6].get_weights())
        input_layer = layers.Input(shape=IMG_SHAPE)
        from_BW = layers.Conv2D(FILTER_DEPTH * config[0], kernel_size=1, strides=1, padding="same", use_bias=True)
        x = from_BW(input_layer)
        from_BW.set_weights(weights[0])
        x = layers.LeakyReLU(0.2)(x)
        x = conv_block(x, config[0], config[1], config[2], config[3], weights=weights[1:])
        for layer in d_layers:
            x = layer(x)
        self.model.discriminator = keras.models.Model(input_layer, x, name="new_discriminator")
        self.model.discriminator.summary()



"""## Train the end-to-end model

"""

# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)
# learning_rate=0.0002, beta_1=0.0, beta_2=0.7233293237969016, epsilon=1e-8
# learning_rate=0.0002, beta_1=0.0, beta_2=0.99, epsilon=1e-8
generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.0, beta_2=0.99, epsilon=1e-8
)

discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.0, beta_2=0.99, epsilon=1e-8
)

if __name__ == "__main__":
    #   Instantiate the customer `GANMonitor` Keras callback.
    cbk = GANMonitor(num_img=6, latent_dim=noise_dim)

    # Instantiate the optimizer for both networks
    # (learning_rate=0.0002, beta_1=0.5 are recommended)


    # Instantiate the WGAN model.
    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=noise_dim,
        discriminator_extra_steps=1,
    )

    # Compile the WGAN model.
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )

    steps_per_epoch = int(math.ceil(len(train_images) / BATCH_SIZE))
    TRANSITION_SPEED = 1 / steps_per_epoch * EPOCHS_PER_SIZE

    # Start training the model.
    history = wgan.fit(train_images, batch_size=BATCH_SIZE, epochs=TOTAL_EPOCHS, callbacks=[cbk])
    #wgan.fit(train_images, batch_size=int(BATCH_SIZE / 2), epochs=epochs, callbacks=[cbk])
    #wgan.fit(train_images, batch_size=int(BATCH_SIZE / 4), epochs=epochs, callbacks=[cbk])
    #wgan.fit(train_images, batch_size=int(BATCH_SIZE / 8), epochs=epochs, callbacks=[cbk])
    #wgan.fit(train_images, batch_size=int(BATCH_SIZE / 16), epochs=epochs, callbacks=[cbk])
