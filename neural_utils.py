from keras.utils.np_utils import to_categorical

def preprocess(X_train, y_train, X_test, y_test, is_cnn=False):
    image_height, image_width = 28, 28
    if is_cnn:
        X_train = X_train.reshape(60000, 28, 28, 1)
        X_test = X_test.reshape(10000, 28, 28, 1)
    else:
        X_train = X_train.reshape(60000, image_height*image_width)
        X_test = X_test.reshape(10000, image_height*image_width)

    print(X_train.shape)
    print(X_test.shape)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255.0
    X_test /= 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    print(y_train.shape)
    print(y_test.shape)

    return (X_train, y_train, X_test, y_test)