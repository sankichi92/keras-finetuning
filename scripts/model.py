from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.regularizers import l2


nb_classes = 200
weight_decay = 1e-2


def create_model():
    base_model = Xception(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(
            nb_classes,
            activation='softmax',
            W_regularizer=l2(weight_decay))(x)

    model = Model(input=base_model.input, output=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model
