"""
Dense UNet++ for small patch retinal vessel segmentation.

- Uses dense blocks and SE blocks for feature refinement.
- Suitable for biomedical image segmentation.

Usage:
    from dense_unet import dense_unet
    model = dense_unet(input_shape=(128, 128, 1))
"""

from tensorflow.keras import layers, models
import tensorflow as tf

def conv_block(x, kernel_size, filters, dropout, batchnorm):
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x

def dense_block(x, num_layers, growth_rate, dropout=None, batchnorm=False):
    for _ in range(num_layers):
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(growth_rate, (3, 3), padding="same")(x)
        if dropout:
            x = tf.keras.layers.Dropout(dropout)(x)
    return x

def transition_down(x, compression, dropout, batchnorm):
    reduced_filters = int(x.shape[-1] * compression)
    x = layers.Conv2D(reduced_filters, kernel_size=1, padding='same')(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x

def transition_up(x, skip_connection):
    x = layers.Conv2DTranspose(skip_connection.shape[-1], kernel_size=2, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip_connection])
    return x

def se_block(input_tensor, ratio=16):
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(channels // ratio, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Multiply()([input_tensor, se])
    return se

def dense_unet(input_shape, growth_rate=32, num_layers_per_block=[4, 5, 7, 10, 12], compression=0.5, dropout=0.2, batchnorm=True):
    inputs = layers.Input(input_shape)
    skip_connections = []
    x = inputs
    for num_layers in num_layers_per_block:
        x = dense_block(x, num_layers, growth_rate, dropout, batchnorm)
        skip_connections.append(x)
        x = transition_down(x, compression, dropout, batchnorm)
    x = dense_block(x, num_layers_per_block[-1], growth_rate, dropout, batchnorm)
    skip_connections = skip_connections[::-1]
    for i, num_layers in enumerate(num_layers_per_block[:-1]):
        x = transition_up(x, skip_connections[i])
        x = dense_block(x, num_layers, growth_rate, dropout, batchnorm)
        x = se_block(x)
    outputs = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model
