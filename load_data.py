import pickle
import numpy
import keras


def load_data(dataset):
    with open(dataset, 'rb') as val_f:
        val_dataset = pickle.load(val_f)
        x_val = val_dataset["x_val"]
        y_val = val_dataset["y_val"]
        x_val = numpy.asarray(x_val,dtype=numpy.float32)
        x_val /= 255
        y_val = keras.utils.to_categorical(y_val)
        return (x_val, y_val)