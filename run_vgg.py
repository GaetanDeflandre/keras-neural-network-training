import numpy as np
from keras.applications import vgg16
from keras.preprocessing import image

#%%
# Initialize the VGG16 model
model = vgg16.VGG16(weights='imagenet')

# Load and resize images
spoon = image.load_img('img/spoon.jpeg', target_size=(224, 224))
fly = image.load_img('img/fly.jpeg', target_size=(224, 224))

# Images to nimpy arrays
spoon_arr = image.img_to_array(spoon)
spoon_arr = np.expand_dims(spoon_arr, axis=0)

fly_arr = image.img_to_array(fly)
fly_arr = np.expand_dims(fly_arr, axis=0)

# Preprocessing the inputs
spoon_arr = vgg16.preprocess_input(spoon_arr)
fly_arr = vgg16.preprocess_input(fly_arr)

#%%
# Predict
spoon_preds = model.predict(spoon_arr)
fly_preds = model.predict(fly_arr)

# Show the predictions
print(vgg16.decode_predictions(spoon_preds, top=5))
print(vgg16.decode_predictions(fly_preds, top=5))