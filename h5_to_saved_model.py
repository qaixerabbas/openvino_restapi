import tensorflow as tf

myModel = tf.keras.models.load_model("bees_keras_model.h5")
tf.saved_model.save(myModel, "myModel")
