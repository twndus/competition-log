import tensorflow as tf

def mlpclassifier_keras(input_shape, learning_rate):

    # build model
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(units=1000, activation='relu')(input_layer)
    x = tf.keras.layers.Dense(units=500, activation='relu')(x)
    x = tf.keras.layers.Dense(units=2, activation='sigmoid')(x)
    model = tf.keras.Model(input_layer, x)

    # optimizer
    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    auroc = tf.keras.metrics.AUC()

    # compile model
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=auroc)

    return model
