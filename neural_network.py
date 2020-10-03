from keras.datasets import mnist
from keras.preprocessing.image import load_img, array_to_img
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt

from neural_utils import preprocess

#%%
# Load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#%%
#plt.imshow(X_train[0], cmap='gray')
#plt.show()
#print(y_train[0])

#%%
# Preprocessing the image data
X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test)

#%%
# Build a model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# %%
# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))


# %%
# Show the accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.show()

# %%
# Evaluating the model
score = model.evaluate(X_test, y_test)