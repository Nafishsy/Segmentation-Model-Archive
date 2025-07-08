from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dense, LayerNormalization, Add, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention

def transformer_block(x, num_heads=4, ff_dim=128):
    projected_x = Dense(ff_dim)(x)
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim // num_heads)(projected_x, projected_x)
    add_skip = Add()([attention, projected_x])
    attention_output = LayerNormalization()(add_skip)
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dense(projected_x.shape[-1])(ff_output)
    add_skip_ff = Add()([ff_output, attention_output])
    return LayerNormalization()(add_skip_ff)

def medt_architecture(input_shape):
    filters = [16, 32, 64, 128, 256]
    inputs = Input(input_shape)
    x = Conv2D(filters[0], 3, padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters[1], 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters[2], 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters[3], 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters[4], 3, padding='same', activation='relu')(x)
    x = transformer_block(x, num_heads=4, ff_dim=filters[4])
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters[3], 3, padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters[2], 3, padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters[1], 3, padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters[0], 3, padding='same', activation='relu')(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    model = Model(inputs, outputs, name="MedT_Architecture")
    return model
