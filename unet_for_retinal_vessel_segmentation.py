from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB4

def AttentionBlock(filters):
    def block(x):
        # Dummy attention block for demonstration
        return x
    return block

def unet_for_retinal_vessel_segmentation(input_shape):
    inputs = Input(shape=input_shape)
    x = Concatenate()([inputs, inputs, inputs])  # Duplicate single channel input to 3 channels
    base_model = EfficientNetB4(input_tensor=x, include_top=False)
    x1 = base_model.get_layer('input_1').output
    x2 = base_model.get_layer('block2a_activation').output
    x3 = base_model.get_layer('block3b_activation').output
    x4 = base_model.get_layer('block4c_activation').output
    x5 = base_model.get_layer('block5a_activation').output
    up6 = UpSampling2D(size=(2, 2))(x5)
    merge6 = Concatenate()([x4, up6])
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.5)(conv6)
    attention6 = AttentionBlock(512)(conv6)
    up7 = UpSampling2D(size=(2, 2))(attention6)
    merge7 = Concatenate()([x3, up7])
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(0.5)(conv7)
    attention7 = AttentionBlock(256)(conv7)
    up8 = UpSampling2D(size=(2, 2))(attention7)
    merge8 = Concatenate()([x2, up8])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(0.5)(conv8)
    attention8 = AttentionBlock(128)(conv8)
    up9 = UpSampling2D(size=(2, 2))(attention8)
    merge9 = Concatenate()([x1, up9])
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(0.5)(conv9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=outputs)
    return model
