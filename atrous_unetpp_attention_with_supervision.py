"""
TransFusion-NetPP (Atrous UNet++ with Attention and Supervision).

- Combines UNet++ skip connections, atrous convolutions, attention, and SE blocks.
- Suitable for retinal vessel segmentation.

Usage:
    from atrous_unetpp_attention_with_supervision import atrous_unetpp_attention_with_supervision
    model = atrous_unetpp_attention_with_supervision(input_shape=(128, 128, 1))
"""

from tensorflow.keras import layers, models

def conv_block(x, kernel_size, filters, dropout, batchnorm):
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x

def se_block(input_tensor, ratio=16):
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(channels // ratio, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Multiply()([input_tensor, se])
    return se

def atrous_unetpp_attention_with_supervision(input_shape, dropout=0.2, batchnorm=True):
    filters = [16, 32, 64, 128, 256]
    kernel_size = 3
    upsample_size = 2

    inputs = layers.Input(input_shape)

    # Downsampling Path
    dn_1 = conv_block(inputs, kernel_size, filters[0], dropout, batchnorm)
    pool_1 = layers.MaxPooling2D(pool_size=(2, 2))(dn_1)

    dn_2 = conv_block(pool_1, kernel_size, filters[1], dropout, batchnorm)
    pool_1_resized = layers.Conv2D(filters[1], (1, 1))(pool_1)
    dn_2 = layers.Add()([dn_2, pool_1_resized])
    pool_2 = layers.MaxPooling2D(pool_size=(2, 2))(dn_2)

    dn_3 = conv_block(pool_2, kernel_size, filters[2], dropout, batchnorm)
    pool_2_resized = layers.Conv2D(filters[2], (1, 1))(pool_2)
    dn_3 = layers.Add()([dn_3, pool_2_resized])
    pool_3 = layers.MaxPooling2D(pool_size=(2, 2))(dn_3)

    dn_4 = conv_block(pool_3, kernel_size, filters[3], dropout, batchnorm)
    pool_3_resized = layers.Conv2D(filters[3], (1, 1))(pool_3)
    dn_4 = layers.Add()([dn_4, pool_3_resized])
    pool_4 = layers.MaxPooling2D(pool_size=(2, 2))(dn_4)

    dn_5 = conv_block(pool_4, kernel_size, filters[4], dropout, batchnorm)

    # Atrous Convolution for Multi-Scale Features
    atrous_1 = layers.Conv2D(filters[4], kernel_size, dilation_rate=1, padding='same')(dn_5)
    atrous_6 = layers.Conv2D(filters[4], kernel_size, dilation_rate=6, padding='same')(dn_5)
    atrous_12 = layers.Conv2D(filters[4], kernel_size, dilation_rate=12, padding='same')(dn_5)
    atrous_18 = layers.Conv2D(filters[4], kernel_size, dilation_rate=18, padding='same')(dn_5)
    atrous_combined = layers.Concatenate()([atrous_1, atrous_6, atrous_12, atrous_18])
    atrous_processed = conv_block(atrous_combined, kernel_size, filters[4], dropout, batchnorm)

    # Upsampling Path with Attention and SE Blocks
    up_5 = layers.UpSampling2D(size=(upsample_size, upsample_size))(atrous_processed)
    dn_4_att = layers.Multiply()([dn_4, layers.Conv2D(filters[3], 1, activation='sigmoid')(dn_4)])
    up_5 = layers.Concatenate()([up_5, dn_4_att])
    up_conv_5 = conv_block(up_5, kernel_size, filters[3], dropout, batchnorm)
    up_conv_5 = se_block(up_conv_5)

    up_4 = layers.UpSampling2D(size=(upsample_size, upsample_size))(up_conv_5)
    dn_3_att = layers.Multiply()([dn_3, layers.Conv2D(filters[2], 1, activation='sigmoid')(dn_3)])
    up_4 = layers.Concatenate()([up_4, dn_3_att])
    up_conv_4 = conv_block(up_4, kernel_size, filters[2], dropout, batchnorm)
    up_conv_4 = se_block(up_conv_4)

    up_3 = layers.UpSampling2D(size=(upsample_size, upsample_size))(up_conv_4)
    dn_2_att = layers.Multiply()([dn_2, layers.Conv2D(filters[1], 1, activation='sigmoid')(dn_2)])
    up_3 = layers.Concatenate()([up_3, dn_2_att])
    up_conv_3 = conv_block(up_3, kernel_size, filters[1], dropout, batchnorm)
    up_conv_3 = se_block(up_conv_3)

    up_2 = layers.UpSampling2D(size=(upsample_size, upsample_size))(up_conv_3)
    dn_1_att = layers.Multiply()([dn_1, layers.Conv2D(filters[0], 1, activation='sigmoid')(dn_1)])
    up_2 = layers.Concatenate()([up_2, dn_1_att])
    up_conv_2 = conv_block(up_2, kernel_size, filters[0], dropout, batchnorm)
    up_conv_2 = se_block(up_conv_2)

    # Output Layer
    conv_final = layers.Conv2D(1, kernel_size=(1, 1))(up_conv_2)
    outputs = layers.Activation('sigmoid')(conv_final)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model
