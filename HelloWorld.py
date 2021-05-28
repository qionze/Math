import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, MaxPool2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import json

batch_size = 128
num_classes = 10
epochs = 10

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(input_shape)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))

model.summary()
#  下面这里是我们的神经网络的模型
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 32)        832
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 28, 28, 32)        25632
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
# _________________________________________________________________
# dropout (Dropout)            (None, 14, 14, 32)        0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 14, 14, 64)        18496
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 14, 14, 64)        36928
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 7, 7, 64)          0
# _________________________________________________________________
# flatten (Flatten)            (None, 3136)              0
# _________________________________________________________________
# dense (Dense)                (None, 256)               803072
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 256)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                2570
# =================================================================

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

model_file = 'model.h5'
model.save(model_file)


