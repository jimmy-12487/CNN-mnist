import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.datasets import mnist
from absl import app
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, Dropout, MaxPool2D
import numpy as np

def main(_agrv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config = config)
		
    (train_image, train_label), (test_image, test_label) = mnist.load_data()
    train_image = train_image.reshape(60000, 28, 28, 1)/255
    test_image = test_image.reshape(10000, 28, 28, 1)/255
    train_label = tf.keras.utils.to_categorical(train_label)
    test_label = tf.keras.utils.to_categorical(test_label)
    batch_size = 128
    input_layer = tf.keras.Input([28, 28, 1])
    
    conv = Conv2D(filters = 64, kernel_size = 8, strides = 1, padding = 'same', activation = 'relu')(input_layer)
    conv = MaxPool2D(pool_size = (2, 2), strides = 1)(conv)
    conv = Conv2D(filters = 128, kernel_size = 6, strides = 2, padding = 'valid', activation = 'relu' )(conv)
    conv = Conv2D(filters = 128, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu' )(conv)
    conv = MaxPool2D(pool_size = (2, 2), strides = 1)(conv)
    conv = Dropout(0.25)(conv)
    conv = Flatten()(conv)
    conv = Dense(128, activation = 'relu')(conv)
    conv = Dense(64, activation = 'relu')(conv)
    conv = Dropout(0.1)(conv)
    conv = Dense(10)(conv)
    print(conv.shape)
    
    Model = tf.keras.Model(input_layer, conv)
    Model.summary()

    learning_rate = 0.0009
    for _ in range(10):
        opt = tf.keras.optimizers.Adam(learning_rate)
        progress_bar_train = tf.keras.utils.Progbar(len(train_image))
        for i in range(0, len(train_image), batch_size):
            with tf.GradientTape() as tape:
                train_images = train_image[i : min(i + batch_size, len(train_image))]
                pred = Model(train_images, training = True)
                pred = tf.reshape(pred, (train_images.shape[0], 10))
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True)(train_label[i: min(i + batch_size, len(train_image))], pred)
                progress_bar_train.add(train_images.shape[0], values = [("loss", loss)])
            gradient = tape.gradient(loss, Model.trainable_variables)
            opt.apply_gradients(zip(gradient, Model.trainable_variables))
        learning_rate *= 0.8
				
    acc = 0
    progress_bar_test = tf.keras.utils.Progbar(len(test_image))
    for i, test in enumerate(test_image):
        pred = Model(test[None, :, :, :])
        pred = tf.reshape(pred, (10, ))
        if(tf.math.argmax(pred) == tf.math.argmax(test_label[i])):
            acc += 1
        progress_bar_test.add(1, values = [("acc", acc/(i+1))])
    print("acc :", acc)
    print(len(test_image))
    print("accuray : ", acc/len(test_image))
		
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
