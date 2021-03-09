from tensorflow.keras import models, layers, optimizers, callbacks, wrappers


def cnn_model(kernel_size=(3, 3), pool_size=(2, 2), first_filters=32, second_filters=64,
              third_filters=128, dropout_conv=0.3, dropout_dense=0.3, image_size=32):
    
    model = models.Sequential()
    model.add(layers.Conv2D(first_filters, kernel_size, activation= 'relu', 
                            input_shape = (image_size, image_size, 3)))
    model.add(layers.MaxPooling2D(pool_size=pool_size))
    model.add(layers.Con2D(first_filters, kernel_size, activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size=pool_size))
    model.add(layers.Con2D(first_filters, kernel_size, activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(7, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4),
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = cnn_model()
    model.summary
