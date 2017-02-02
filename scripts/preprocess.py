from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


img_width, img_height = 299, 299


def data_generator(directory, augmentation=False, shuffle=True):
    if augmentation is True:
        datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')
    else:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    generator = datagen.flow_from_directory(
            directory,
            target_size=(img_height, img_width),
            shuffle=shuffle)

    return generator
