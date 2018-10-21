import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import pickle
from matplotlib import pyplot as plt

from pymongo import MongoClient
client = MongoClient()
db = client.get_database("keras")
res_col = db.get_collection("dpm")


def smooth_curve(data):
    smoothed_data = []
    for i in range(len(data)):
        if i == 0:
            smoothed_data.append((data[i]*2 + data[i+1])/3)
        elif i == len(data)-1:
            smoothed_data.append((data[i]*2 + data[i-1])/3)
        else:
            smoothed_data.append((data[i-1] + data[i] + data[i+1])/3)
    return smoothed_data


def plot_res(hists):
    colors = ['b', 'r', 'g', 'c', 'y', 'k', 'm']

    plt.figure(figsize=(20, 15))
    i = 0
    for hist_name in hists.keys():
        hist = hists[hist_name]
        epoches = len(hist.history['acc'])
        plt.plot(range(epoches), smooth_curve(hist.history['acc']), '%so'%colors[i], label="%s_acc"%hist_name)
        plt.plot(range(epoches), smooth_curve(hist.history['val_acc']), "%s"%colors[i], label='%s_val_acc'%hist_name)
        i += 1
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("acc")

    plt.figure(figsize=(20, 15))
    i = 0
    for hist_name in hists.keys():
        hist = hists[hist_name]
        epoches = len(hist.history['loss'])
        plt.plot(range(epoches), smooth_curve(hist.history['loss']), '%so'%colors[i], label="%s_acc"%hist_name)
        plt.plot(range(epoches), smooth_curve(hist.history['val_loss']), "%s"%colors[i], label='%s_val_loss'%hist_name)
        i += 1
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss")

    plt.show()


def extract_features(model, directory, batch_size):
    # 利用预训练网络提取特征
    import os
    sample_count = sum(len(os.listdir(os.path.join(directory, subdir))) for subdir in os.listdir(directory))

    feature_shape = model.output.shape[1:]
    print(feature_shape)
    input_shape = model.input_shape[1:3]
    print(model.input_shape)
    features = np.zeros((sample_count, feature_shape[0], feature_shape[1], feature_shape[2]))
    labels = np.zeros(shape=(sample_count, len(os.listdir(directory))))
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        directory,
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical')
    i = 0

    for input_batch, labels_batch in generator:
        features_batch = model.predict(input_batch)
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        i += 1
        print("第%s个" % (i * batch_size))
        if i * batch_size >= sample_count:
            break
    return features, labels


def generate_pkz_dataset():
    target_dir = r"G:\pic_data\dpm\origin\type23"
    model_name = "vgg16"
    dataset_name = "dpm_23"
    input_shape = 224
    f_pkz_base = os.path.join(r"F:\jupyter-notebook\deeplearning\features",
                              "{}_{}_{}".format(model_name, dataset_name, input_shape))
    print(f_pkz_base)

    print("extract...")
    x, y = extract_features(
        keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(input_shape, input_shape, 3)),
        target_dir, 16)

    print("dump...")
    batch_pkz = 2000
    i = 0
    while i * batch_pkz < len(x):
        x_batch = x[i * batch_pkz: (i + 1) * batch_pkz]
        y_batch = y[i * batch_pkz: (i + 1) * batch_pkz]
        with open(r"{}_batch{}".format(f_pkz_base, i + 1), 'wb') as f:
            pickle.dump({"x": x_batch, "y": y_batch}, f)
        i += 1

    print(f_pkz_base, "完成")


def get_generator(pkzs, batch_size):
    i = 0
    j = 0
    x, y = [], []
    while True:
        if len(x) < i + batch_size:
            if j == len(pkzs):
                j = 0
            f_pkz = pkzs[j]
            with open(f_pkz, 'rb') as f:
                data = pickle.load(f)
                print("load ", f_pkz, len(data['x']))
                j += 1
            if len(x) == 0:
                x = data['x']
                y = data['y']
                print("init x", len(x))
            else:
                x = np.concatenate((x[i:], data['x']))
                y = np.concatenate((y[i:], data['y']))
                print("new x", len(x))
                i = 0
        x_batch = x[i:i+batch_size]
        x_batch = np.reshape(x_batch, (len(x_batch), -1))
        y_batch = y[i:i+batch_size]
        yield x_batch, y_batch
        i += batch_size





def get_dense_model1(nb_class, input_dim):
    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=input_dim, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(nb_class, activation='softmax'))
    return model


def get_dense_model2(nb_class, input_dim):
    model = models.Sequential()
    model.add(layers.Dense(512, input_dim=input_dim, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(nb_class, activation='softmax'))
    return model

def get_dense_model3(nb_class, input_dim):
    model = models.Sequential()
    model.add(layers.Dense(512, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(nb_class, activation='softmax'))
    return model

def get_dense_model4(nb_class, input_dim):
    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(nb_class, activation='softmax'))
    return model


def get_dense_model5(nb_class, input_dim):
    model = models.Sequential()
    model.add(layers.Dense(512, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(nb_class, activation='softmax'))
    return model


def run_model_from_memory(f_pkz_base, nb_train):
    pkz_dir = r"F:\jupyter-notebook\deeplearning\features"
    # f_pkz_base = r"{}_dpm_10_224".format(model)
    f_pkzs = [os.path.join(pkz_dir, x) for x in os.listdir(pkz_dir) if x.startswith(f_pkz_base)]
    x, y = None, None
    for f_pkz in f_pkzs:
        print("load ", f_pkz)
        with open(f_pkz, 'rb') as f:
            data_batch = pickle.load(f)
        if x == None:
            x = data_batch["x"]
            y = data_batch["y"]
        else:
            x = np.concatenate((x, data_batch["x"]))
            y = np.concatenate((y, data_batch["y"]))
    print(x.shape, y.shape)
    X = np.reshape(x, (x.shape[0], -1))
    Y = y

    # slices = np.arange(len(X))
    # np.random.shuffle(slices)
    # X = X[slices]
    # Y = y[slices]

    import copy
    z =copy.copy(y[nb_train:])
    acc = []
    for i in range(10):
        np.random.shuffle(z)
        res = [1 if (z[j] == y[j+nb_train]).all() else 0 for j in range(len(z))]
        acc.append(sum(res)/len(res))
    print("基线测试", sum(acc)/len(acc))

    model = get_dense_model1(23, X.shape[1])
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss=losses.categorical_crossentropy, metrics=['acc'])

    hist = model.fit(
        X[:nb_train],
        Y[:nb_train],
        epochs=50,
        batch_size=32,
        validation_data=(X[nb_train:], Y[nb_train:]),
        verbose=1)

    res_col.insert_one({
        "features": f_pkz_base,
        "hist": hist.history,
        "train": nb_train,
        "val": len(X) - nb_train,
        "model": "get_dense_model1",
        "comment": "dpm_23_inceptionv3_lr1_4"
    })


def run_model_from_disk(model):
    pkz_dir = r"F:\jupyter-notebook\deeplearning\features"
    f_pkz_base = r"{}_dpm_10_224".format(model)
    f_pkzs = [os.path.join(pkz_dir, x) for x in os.listdir(pkz_dir) if x.startswith(f_pkz_base)]
    model = get_dense_model3(10, 7*7*2048)
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss=losses.categorical_crossentropy, metrics=['acc'])
    train_pkzs = f_pkzs[:3]
    val_pkzs = f_pkzs[3:]
    hist = model.fit_generator(
        get_generator(train_pkzs, 32),
        epochs=50,
        steps_per_epoch=93,
        validation_data=get_generator(val_pkzs, 32),
        validation_steps=62,
        verbose=2)

    res_col.insert_one({
        "features": f_pkz_base,
        "hist": hist.history,
        "train": 3000,
        "val": 2147,
        "model": "get_dense_model3"
    })


def test_gen():

    pkz_dir = r"F:\jupyter-notebook\deeplearning\features"
    f_pkz_base = r"xception_dpm_10_224"
    f_pkzs = [os.path.join(pkz_dir, x) for x in os.listdir(pkz_dir) if x.startswith(f_pkz_base)]

    train_pkzs = f_pkzs[:3]
    val_pkzs = f_pkzs[3:]
    gen = get_generator(train_pkzs, 32)
    count = 0
    offset = 100
    X1 = None
    while True:
        x1, y1 = next(gen)
        print(x1.shape, y1.shape)
        count += len(x1)
        print(count)
        if count >= offset*32:
            X1 = x1
            break
    print(x1.shape)

    x, y = [], []
    for f_pkz in f_pkzs[:3]:
        print("load ", f_pkz)
        with open(f_pkz, 'rb') as f:
            data_batch = pickle.load(f)
            print(len(data_batch['x']))
        if len(x) == 0:
            x = data_batch["x"]
            y = data_batch["y"]
            print("init x", len(x))
        else:
            x = np.concatenate((x, data_batch["x"]))
            y = np.concatenate((y, data_batch["y"]))
            print("new x", len(x))

    print(x.shape)
    x = np.reshape(x, (len(x), -1))
    print((X1 == x[(32*offset % 3000) - 32: 32*offset % 3000]).all())


if __name__ == '__main__':
    # for model in ["vgg16","vgg19","resnet50","inceptionv3"]:
    #     run_model(model)
    run_model_from_memory("vgg16_dpm_23_224", 4000)
    # generate_pkz_dataset()