import numpy as np

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input

#%%
# Initialize the image data generators

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

jf_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True
)

#%%
# Generate images for the data sets
train_generator = train_datagen.flow_from_directory('img/sample-train/', target_size=(150, 150), save_to_dir='img/sample-confirm/')

i=0
for batch in train_datagen.flow_from_directory('img/sample-train/', target_size=(150,150), save_to_dir='img/sample-confirm/'):
    i+=1
    if (i>10):
        break

j=0
for batch in jf_datagen.flow_from_directory('img/sample-train/', target_size=(150,150), save_to_dir='img/sample-confirm/'):
    j+=1
    if ( j > 10):
        break