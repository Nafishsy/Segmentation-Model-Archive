from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Multiply, Add
from tensorflow.keras.models import Model

def attention_gate(x, g, filters):
    g1 = Conv2D(filters, kernel_size=1, activation='relu')(g)
    x1 = Conv2D(filters, kernel_size=1, activation='relu')(x)
    psi = Add()([g1, x1])
    psi = Conv2D(1, kernel_size=1, activation='sigmoid')(psi)
    return Multiply()([x, psi])

def multi_scale_features(x, filters):
    scales = []
    for rate in [1, 2, 3]:
        scaled = Conv2D(filters, kernel_size=3, dilation_rate=rate, padding='same', activation='relu')(x)
        scales.append(scaled)
    return Concatenate()(scales)

def enhanced_segnet2(input_shape):
    filters = [8, 16, 32, 64, 128]
    inputs = Input(input_shape)
    enc1 = multi_scale_features(inputs, filters[0])
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(enc1)
    enc2 = multi_scale_features(pool1, filters[1])
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(enc2)
    enc3 = multi_scale_features(pool2, filters[2])
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(enc3)
    enc4 = multi_scale_features(pool3, filters[3])
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(enc4)
    bottleneck = multi_scale_features(pool4, filters[4])
    up4 = UpSampling2D(size=(2, 2))(bottleneck)
    att4 = attention_gate(enc4, up4, filters[3])
    dec4 = Conv2D(filters[3], kernel_size=3, padding='same', activation='relu')(Concatenate()([up4, att4]))
    up3 = UpSampling2D(size=(2, 2))(dec4)
    att3 = attention_gate(enc3, up3, filters[2])
    dec3 = Conv2D(filters[2], kernel_size=3, padding='same', activation='relu')(Concatenate()([up3, att3]))
    up2 = UpSampling2D(size=(2, 2))(dec3)
    att2 = attention_gate(enc2, up2, filters[1])
    dec2 = Conv2D(filters[1], kernel_size=3, padding='same', activation='relu')(Concatenate()([up2, att2]))
    up1 = UpSampling2D(size=(2, 2))(dec2)
    att1 = attention_gate(enc1, up1, filters[0])
    dec1 = Conv2D(filters[0], kernel_size=3, padding='same', activation='relu')(Concatenate()([up1, att1]))
    outputs = Conv2D(1, kernel_size=1, activation='sigmoid')(dec1)
    model = Model(inputs, outputs, name="Enhanced_SegNet2")
    return model
