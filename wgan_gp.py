# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend
import gc

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Define growth of resolutions
img_shape_config = [
    (4, 4, 1),
    (8, 8, 1),
    (16, 16, 1),
    (32, 32, 1),
    (64, 64, 1),
    (128, 128, 1),
    (256, 256, 1)
]

# Current resolution
IMG_SHAPE = img_shape_config[0]
# Epochs per resolution
EPOCHS_PER_SIZE = 50
# Total number of epochs (when using progressive growth)
TOTAL_EPOCHS = EPOCHS_PER_SIZE * ((len(img_shape_config)-1) * 2 + 1)
# Size of Minibatches
BATCH_SIZE = 32
# Discriminator training runs per generator training run
N_DISCRIMINATOR = 5
# Current state of transition. 1.0 in stabilization phases
CURRENT_TRANSITION = tf.Variable(1.0, trainable=False)
# Transition growth per minibatch, gets calculated automatically
TRANSITION_SPEED = 0.0
# Returns True when in transition phase
IS_TRANSITION = False
# Enables progressive growth when True
DO_GROWTH = False
# Saves models after each epoch when True
DO_SAVE = True

# Size/Dimension of the latent noise of the generator
noise_dim = 512

# When not progressively growing, set resolution
if not DO_GROWTH:
    IMG_SHAPE = (256, 256, 1)


# Method for loading images from storage into numpy arrays for training
# res is a int and determines the resolution of the loaded images
def get_images(res):
    images = []
    # Iterate over 86176 training images
    for index in range(86176):
        # Build path and load the image as a grayscale image
        image = cv2.imread("data\\img\\" + str(index) + ".png", cv2.IMREAD_GRAYSCALE)
        # Resize image when needed
        if res != 256:
            image = cv2.resize(image, (res, res))
        # Add images to list
        images.append(image)
        del image

    # Convert list to array
    images = np.array(images, dtype="float32")
    # Reshape array for training
    images = images.reshape(images.shape[0], *(res, res, 1))
    # Print shape
    print(images.shape)
    # Scale images for training, so the generator trains for outputs between -1 and 1
    images = (images - 127.5) / 127.5
    # Shuffle images
    np.random.seed(3213)
    np.random.shuffle(images)
    return images

# Load images
train_images = get_images(IMG_SHAPE[0])

# Pixel normalization "layer"
# src: https://github.com/tkarras/progressive_growing_of_gans/blob/6c2a3add95ce0d12e0df089cacbb3d4a56cec019/networks.py#L120
def pixel_norm(x, epsilon=1e-8):
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

# Linear interpolation function, transitions between two tensors based on weight t
# src: https://github.com/tkarras/progressive_growing_of_gans/blob/6c2a3add95ce0d12e0df089cacbb3d4a56cec019/networks.py#L15
def lerp(a, b, t):
    return a + (b - a) * t

# Equalized Convolutional Layer
# Scales weights (i.e. the kernel) with he constant
# src: https://github.com/p4vv37/3dml/blob/d0096bca0c487de2b4408d4ac6807ac3576fea9d/5_generative_networks/progan/layers.py#L111
class EqualizedConv2D(layers.Conv2D):
    def __init__(self, *args, **kwargs):
        self.scale = 1.0
        super(EqualizedConv2D, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        fan_in = np.product([int(val) for val in input_shape[1:]])
        self.scale = np.sqrt(2/fan_in)
        return super(EqualizedConv2D, self).build(input_shape)

    def call(self, inputs):
        outputs = backend.conv2d(inputs, self.kernel * self.scale, strides=self.strides, padding=self.padding,
                                 data_format=self.data_format, dilation_rate=self.dilation_rate)
        if not DO_GROWTH:
            outputs = backend.conv2d(inputs, self.kernel, strides=self.strides, padding=self.padding, data_format=self.data_format, dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = backend.bias_add(outputs, self.bias, data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

# Equalized Dense Layer
# Same effect as EqualizedConv2D for dense layer
# Functionality gets disabled when not progressively growing
class EqualizedDense(layers.Dense):
    def __init__(self, *args, gain=1, **kwargs):
        self.scale = 1.0
        self.gain = gain
        super(EqualizedDense, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        fan_in = np.product([int(val) for val in input_shape[1:]])
        self.scale = self.gain / np.sqrt(fan_in)
        return super(EqualizedDense, self).build(input_shape)

    def call(self, inputs):
        outputs = backend.dot(inputs, self.kernel*self.scale)
        if not DO_GROWTH:
            outputs = backend.dot(inputs, self.kernel)
        if self.use_bias:
            outputs = backend.bias_add(outputs, self.bias, data_format='channels_last')
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

# Minibatch standard deviation layer
# src: https://github.com/p4vv37/3dml/blob/d0096bca0c487de2b4408d4ac6807ac3576fea9d/5_generative_networks/progan/layers.py#L35
class MinibatchStdev(layers.Layer):
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mean = backend.mean(inputs, axis=0, keepdims=True)
        squ_diffs = backend.square(inputs - mean)
        mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
        mean_sq_diff += 1e-8
        stdev = backend.sqrt(mean_sq_diff)
        mean_pix = backend.mean(stdev, keepdims=True)
        shape = backend.shape(inputs)
        output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        combined = backend.concatenate([inputs, output], axis=-1)
        return combined

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)

# Initializer for Equalized Learning
initializer = tf.keras.initializers.RandomNormal(0, 1)

# Change initializer and batch size when not using progressive growth
if not DO_GROWTH:
    initializer = tf.keras.initializers.GlorotUniform()
    BATCH_SIZE = 16

# Convolutional/Downsampling block of discriminator
def conv_block(x, depth, depth_multiplier=2, weights=None, downsample=True):
    l_1 = EqualizedConv2D(depth, kernel_size=3, kernel_initializer=initializer, strides=1, padding="same")
    x = l_1(x)
    if weights:
        l_1.set_weights(weights[0])
    x = layers.LeakyReLU(0.2)(x)
    if downsample:
        l_2 = EqualizedConv2D(depth * depth_multiplier, kernel_size=3,
                              kernel_initializer=initializer, strides=1, padding="same")
        x = l_2(x)
        if weights:
            l_2.set_weights(weights[1])
        x = layers.LeakyReLU(0.2)(x)
        x = layers.AveragePooling2D(2)(x)
    return x

# Transition block of discriminator
def d_transition_block(x, depth, depth_multiplier=2, weights=None):
    y = layers.AveragePooling2D(2)(x)
    old_exit = EqualizedConv2D(depth * depth_multiplier, kernel_size=1, kernel_initializer=initializer, strides=1, padding="same", trainable=False)
    y = old_exit(y)
    if weights:
        old_exit.set_weights(weights)
    y = layers.LeakyReLU(0.2)(y)

    x = EqualizedConv2D(depth, kernel_size=1, kernel_initializer=initializer, strides=1, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = conv_block(x, depth, depth_multiplier)
    # Interpolate between old and new output
    x = lerp(y, x, CURRENT_TRANSITION)
    return x

# Filter sizes of convolutional blocks of the discriminator
d_layer_config = [
    [512, 1],
    [512, 1],
    [256, 2],
    [128, 2],
    [64, 2],
    [32, 2],
    [16, 2]
]


# Discriminator model
def get_discriminator_model():
    img_input = layers.Input(shape=IMG_SHAPE)
    # Build whole network when not progressively growing
    if not DO_GROWTH:
        x = EqualizedConv2D(d_layer_config[-1][0], kernel_size=1, kernel_initializer=initializer, strides=1, padding="same")(
            img_input)
        x = layers.LeakyReLU(0.2)(x)
        x = conv_block(x, *d_layer_config[6])
        x = conv_block(x, *d_layer_config[5])
        x = conv_block(x, *d_layer_config[4])
        x = conv_block(x, *d_layer_config[3])
        x = conv_block(x, *d_layer_config[2])
        x = conv_block(x, *d_layer_config[1])
    else:
        x = EqualizedConv2D(d_layer_config[0][0], kernel_size=1, kernel_initializer=initializer, strides=1,
                            padding="same")(
            img_input)
        x = layers.LeakyReLU(0.2)(x)
    # Disable minibatch standard deviation layer when not progressivly growing
    if DO_GROWTH:
        x = MinibatchStdev()(x)

    x = conv_block(x, *d_layer_config[0], downsample=False)

    x = layers.Flatten()(x)
    if DO_GROWTH:
        x = EqualizedDense(d_layer_config[0][0], gain=np.sqrt(2))(x)
    else:
        x = layers.Dense(d_layer_config[0][0])(x)

    x = layers.LeakyReLU(0.2)(x)

    if DO_GROWTH:
        x = EqualizedDense(1, gain=1)(x)
    else:
        x = layers.Dense(1)(x)

    d_model = keras.models.Model(img_input, x, name="discriminator")
    return d_model


# Deconvolutional/Upsampling block of the generator
def deconv_block(x, depth, upsampling=True):
    if upsampling:
        x = layers.UpSampling2D(2, interpolation="nearest")(x)
        x = EqualizedConv2D(depth, kernel_size=3, kernel_initializer=initializer, strides=1, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = pixel_norm(x)
    x = EqualizedConv2D(depth, kernel_size=3, kernel_initializer=initializer, strides=1, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = pixel_norm(x)
    return x

# Transition block of the generator
def g_transition_block(x, depth, weights=None):
    old_exit = EqualizedConv2D(filters=IMG_SHAPE[-1], kernel_size=1, kernel_initializer=initializer, strides=1, padding="same", trainable=False)
    y = old_exit(x)
    if weights:
        old_exit.set_weights(weights)
    y = layers.UpSampling2D(2, interpolation="nearest")(y)
    y = layers.Activation("linear")(y)

    x = deconv_block(x, depth)
    x = EqualizedConv2D(filters=IMG_SHAPE[-1], kernel_size=1, kernel_initializer=initializer, strides=1, padding="same")(x)
    x = layers.Activation("linear")(x)
    # Interpolate between old and new output
    x = lerp(y, x, CURRENT_TRANSITION)
    return x

# Filter sizes of deconvolutional blocks of the generator
g_layer_config = [
    [512],
    [512],
    [256],
    [128],
    [64],
    [32],
    [16]
]

# Generator model
def get_generator_model():
    noise = layers.Input(shape=(noise_dim,))
    if DO_GROWTH:
        x = EqualizedDense(g_layer_config[0][0] * 4 * 4, gain=np.sqrt(2)/4)(noise)
    else:
        x = layers.Dense(g_layer_config[0][0] * 4 * 4)(noise)

    x = layers.Reshape((4, 4, g_layer_config[0][0]))(x)

    x = deconv_block(x, *g_layer_config[0], upsampling=False)
    # Build whole model when not progressively growing
    if not DO_GROWTH:
        x = deconv_block(x, *g_layer_config[1])
        x = deconv_block(x, *g_layer_config[2])
        x = deconv_block(x, *g_layer_config[3])
        x = deconv_block(x, *g_layer_config[4])
        x = deconv_block(x, *g_layer_config[5])
        x = deconv_block(x, *g_layer_config[6])

    x = EqualizedConv2D(filters=IMG_SHAPE[-1], kernel_size=1, kernel_initializer=initializer, strides=1, padding="same")(x)
    x = layers.Activation("linear")(x)

    g_model = keras.models.Model(noise, x, name="generator")
    return g_model

# Build models and prints them to the console
if __name__ == "__main__":
    d_model = get_discriminator_model()
    d_model.summary()
    g_model = get_generator_model()
    g_model.summary()

# Wasserstein loss function for discriminator
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

# Generator loss function
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

# Wasserstein GAN with gradient penalty training model
# src: https://github.com/keras-team/keras-io/blob/master/examples/generative/ipynb/wgan_gp.ipynb
class WGAN(keras.Model):
    def __init__(
            self,
            discriminator,
            generator,
            latent_dim,
            discriminator_extra_steps=3,
            gp_weight=10
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    # Compiles the models
    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    # Calculates the gradient penalty
    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    # Training implementation
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        batch_size = tf.shape(real_images)[0]

        # When in transition phase, interpolates the images with upscaled lower resolution ones,
        # to smoothly scale the output resolution
        if IS_TRANSITION:
            small_images = tf.image.resize(real_images, (int(IMG_SHAPE[0]/2), int(IMG_SHAPE[1]/2)))
            small_images = tf.image.resize(small_images, IMG_SHAPE[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            real_images = lerp(small_images, real_images, CURRENT_TRANSITION)

        # Train the discriminator N_DISCRIMINATOR times
        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)

                # Calculate Wasserstein loss
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate and add gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_cost + gp * self.gp_weight
                # Calculae and add epsilon penalty
                epsilon_penalty = tf.square(real_logits)
                d_loss += tf.reduce_mean(epsilon_penalty) * 0.001

            # Apply gradients to discriminator
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Generate random latent noise
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Generate generator loss
            g_loss = self.g_loss_fn(gen_img_logits)
        # Apply gradients to generator
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


# Callback class, handles growth and images
class GANMonitor(keras.callbacks.Callback):

    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim), seed=3342934)
        self.stage = 1

    # Call to set to next size stage
    def next(self):
        self.stage += 1

    # Progressively grows CURRENT_TRANSITON when in transition phase
    def on_train_batch_end(self, step, logs=None):
        if IS_TRANSITION:
            CURRENT_TRANSITION.assign_add(TRANSITION_SPEED)
            if CURRENT_TRANSITION > 1.0:
                CURRENT_TRANSITION.assign(1.0)

    # Creates images and changes networks after the last epoch of a phase
    def on_epoch_end(self, epoch, logs=None):
        global IS_TRANSITION
        generated_images = self.model.generator(self.random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5
        current_epoch = epoch + (self.stage - 1) * EPOCHS_PER_SIZE

        # Generates and saves images to allow track generator learning process
        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.preprocessing.image.array_to_img(img)
            img.save("output\\{epoch}_heightmap_{i}.png".format(i=i, epoch=current_epoch))

        # Save models, only possible when not in transition phase
        if DO_SAVE and not IS_TRANSITION:
            self.model.generator.save("models\\Generator_{epoch}.h5".format(epoch=current_epoch))
            self.model.discriminator.save("models\\Discriminator.h5")

        # Do not change networks when not progressively growing
        if not DO_GROWTH:
            return

        # Return when first epoch, except when EPOCH_PER_SIZE is 1
        if epoch == 0 and EPOCHS_PER_SIZE != 1:
            return

        # If last epoch in phase, change networks
        if epoch == 0 or epoch % (EPOCHS_PER_SIZE - 1) == 0:
            current_network = int(math.ceil(self.stage / 2))
            print("Changing networks on stage " + str(current_network))

            # Change networks to stabilization phase
            if self.stage % 2 == 0:
                print("Changing to non-transition")
                IS_TRANSITION = False
                CURRENT_TRANSITION.assign(1.0)
                self.transition_discriminator(current_network)
                self.transition_generator(current_network)
            # Change networks to transition phase
            else:
                print("Changing to transition")
                IS_TRANSITION = True
                global IMG_SHAPE
                IMG_SHAPE = img_shape_config[current_network]
                CURRENT_TRANSITION.assign(0.0)
                self.extend_discriminator(current_network)
                self.extend_generator(current_network)

    # Changes generator to transition phase by adding transition block
    def extend_generator(self, config):
        config = g_layer_config[config]
        weights = self.model.generator.layers[-2].get_weights()
        self.model.generator = keras.models.Model(self.model.generator.input, g_transition_block(self.model.generator.layers[-3].output, *config, weights=weights), name="new_generator")
        self.model.generator.summary()

    # Changes generator from transition phase to stabilization phase by removing unneeded old layers
    def transition_generator(self, config):
        self.model.generator = keras.models.Model(self.model.generator.input, self.model.generator.layers[-4].output, name="new_generator")
        self.model.generator.summary()

    # Changes discriminator to discriminator phase by adding transition block
    def extend_discriminator(self, config):
        config = d_layer_config[config]
        weights = self.model.discriminator.layers[1].get_weights()
        d_layers = self.model.discriminator.layers[3:]
        input_layer = layers.Input(shape=IMG_SHAPE)
        x = d_transition_block(input_layer, *config, weights)
        for layer in d_layers:
            weight = layer.get_weights()
            x = layer(x)
            layer.set_weights(weight)
        self.model.discriminator = keras.models.Model(input_layer, x, name="new_discriminator")
        self.model.discriminator.summary()

    # Changes discriminator from transition phase to stabilization phase by removing unneeded old layers
    def transition_discriminator(self, config):
        config = d_layer_config[config]
        d_layers = self.model.discriminator.layers[14:]
        weights = []
        weights.append(self.model.discriminator.layers[1].get_weights())
        weights.append(self.model.discriminator.layers[3].get_weights())
        weights.append(self.model.discriminator.layers[6].get_weights())
        input_layer = layers.Input(shape=IMG_SHAPE)
        from_BW = EqualizedConv2D(config[0], kernel_size=1, kernel_initializer=initializer, strides=1, padding="same")
        x = from_BW(input_layer)
        from_BW.set_weights(weights[0])
        x = layers.LeakyReLU(0.2)(x)
        x = conv_block(x, *config, weights=weights[1:])
        for layer in d_layers:
            weight = layer.get_weights()
            x = layer(x)
            layer.set_weights(weight)
        self.model.discriminator = keras.models.Model(input_layer, x, name="new_discriminator")
        self.model.discriminator.summary()


# Optimizer for generator
generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.0, beta_2=0.99, epsilon=1e-8
)

# Optimizer for discriminator
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.0, beta_2=0.99, epsilon=1e-8
)

if __name__ == "__main__":
    # Create GANMonitor
    cbk = GANMonitor(num_img=1, latent_dim=noise_dim)

    # Create WGAN
    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=noise_dim,
        discriminator_extra_steps=N_DISCRIMINATOR
    )

    # Compile WGAN
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss
    )

    #Calculate steps per epoch and TRANSITION_SPEED
    steps_per_epoch = int(math.ceil(len(train_images) / BATCH_SIZE))
    TRANSITION_SPEED = 1 / (steps_per_epoch * EPOCHS_PER_SIZE)

    # When not using progressive growth, fit model once
    if not DO_GROWTH:
        wgan.fit(train_images, batch_size=BATCH_SIZE, epochs=EPOCHS_PER_SIZE, callbacks=[cbk])
    # When using progressive growth, each phase gets own fit call
    # Otherwise the model will not be able to train correctly
    else:
        # Start training the model.
        for i in range(int(TOTAL_EPOCHS / EPOCHS_PER_SIZE)):
            if IS_TRANSITION:
                # Reload images in higher resolution
                del train_images
                gc.collect()
                train_images = get_images(IMG_SHAPE[0])
                # Lower BATCH_SIZE to not exceed vram limitations on higher resolutions
                if cbk.stage == 12 or not DO_GROWTH:
                    BATCH_SIZE = 8
                # Recalculate TRANSITION speed for new batchsize
                steps_per_epoch = int(math.ceil(len(train_images) / BATCH_SIZE))
                TRANSITION_SPEED = 1 / (steps_per_epoch * EPOCHS_PER_SIZE)
            wgan.fit(train_images, batch_size=BATCH_SIZE, epochs=EPOCHS_PER_SIZE, callbacks=[cbk])

            # Model needs to be recompiled to be able to train the added layers
            wgan = WGAN(discriminator=wgan.discriminator, generator=wgan.generator, latent_dim=noise_dim, discriminator_extra_steps=N_DISCRIMINATOR)
            wgan.compile(
                d_optimizer=discriminator_optimizer,
                g_optimizer=generator_optimizer,
                g_loss_fn=generator_loss,
                d_loss_fn=discriminator_loss
            )
            cbk.next()
