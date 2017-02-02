from keras.optimizers import SGD
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from preprocess import data_generator


train_data_dir = '../data/train'
val_data_dir = '../data/validation'

nb_train_samples = 9000
nb_val_samples = 1000

nb_freezed_layers = 115
learning_rate = 5e-4

nb_epoch = 100
patience = 20
monitor = 'val_loss'


def train_top(model, output_path=None):
    model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    train_gen = data_generator(train_data_dir)
    val_gen = data_generator(val_data_dir)

    model.fit_generator(
            train_gen,
            samples_per_epoch=nb_train_samples,
            nb_epoch=1,
            validation_data=val_gen,
            nb_val_samples=nb_val_samples)

    if output_path is not None:
        model.save(output_path)

    return model


def finetune(model, output_path, log_dir):
    for layer in model.layers[:nb_freezed_layers]:
        layer.trainable = False
    for layer in model.layers[nb_freezed_layers:]:
        layer.trainable = True

    model.compile(
            optimizer=SGD(lr=learning_rate, momentum=0.9),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    train_gen = data_generator(train_data_dir, augmentation=True)
    val_gen = data_generator(val_data_dir)

    tb = TensorBoard(log_dir=log_dir, write_graph=False)
    es = EarlyStopping(monitor=monitor, patience=patiance)
    mc = ModelCheckpoint(output_path, monitor=monitor, save_best_only=True)

    model.fit_generator(
            train_gen,
            samples_per_epoch=nb_train_samples,
            nb_epoch=nb_epoch,
            callbacks=[tb, es, mc],
            validation_data=val_gen,
            nb_val_samples=nb_val_samples)

    return model


if __name__ == "__main__":
    import os.mkdir
    from datetime import datetime
    import numpy as np
    from model import create_model

    np.random.seed(sum(map(ord, 'keras-finetuning')))

    now = datetime.now().strftime('%y%m%d-%H%M')
    model_path = '../models/' + now + '.h5'
    log_dir = '../logs/' + now
    os.mkdir(log_dir)

    model = create_model()
    model = train_top(model)
    # weights_path = sys.argv[1]
    # model.load_weights(weights_path)
    finetune(model, model_path, log_dir)
