from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.datasets import mnist

import matplotlib.pyplot  as plt

from neural_utils import preprocess

#%%
# Load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#%%
# Preprocessing the image data
X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test, is_cnn=True)

#%%
# Build the CNN model
cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(5, 5), input_shape=(28, 28, 1), padding='same', activation='relu'))
cnn.add(MaxPooling2D())
cnn.add(Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu'))
cnn.add(MaxPooling2D())
cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dense(10, activation='softmax'))

# Compile the model
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()

# %%
# Train the model
history = cnn.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))


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
score = cnn.evaluate(X_test, y_test)