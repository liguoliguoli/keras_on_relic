import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.applications.xception import Xception,preprocess_input
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.applications.vgg19 import VGG19,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad
from keras.models import load_model
from keras.callbacks import TensorBoard
import os
import time
import sys
import logging
from pymongo import MongoClient


import pickle
import numpy
def load_data(dataset):
    with open(dataset, 'rb') as val_f:
        val_dataset = pickle.load(val_f)
        x_val = val_dataset["x_val"]
        y_val = val_dataset["y_val"]
        x_val = numpy.asarray(x_val,dtype=numpy.float32)
        x_val /= 255
        y_val = [i-1 for i in y_val]
        y_val = keras.utils.to_categorical(y_val)
        return (x_val, y_val)

dataset = r"G:\pic_data\dpm\dataset\array\dataset_dpm_type_nb_class_10_split_6_shape_299_dtype_int8_val"
x_val, y_val = load_data(dataset)
print(x_val.shape[0])

"""
logger配置
"""
logger = logging.getLogger("keras_dpm")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fh = logging.FileHandler("keras.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s -%(name)s-%(levelname)s-%(module)s:%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

"""
可视化配置
"""
tb = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 write_graph=False,  # 是否存储网络结构图
                 write_grads=False, # 是否可视化梯度直方图
                 write_images=False,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)


callbacks = [tb]


def getNum(dir):
    count = 0
    for child in os.listdir(dir):
        subdir = os.path.join(dir,child)
        count += len(os.listdir(subdir))
    return count


def setup_to_transfer_learning(model, base_model):#base_model
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def setup_to_fine_tune(model, GAP_LAYER):
    for layer in model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=Adagrad(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


def learn(cfg_idx=-1,options={}):
    """
    数据库配置
    """
    client = MongoClient()
    db_name = "keras"
    cfg_col_name = "experiment_cfg"
    res_col_name = "experiment_res"
    db = client.get_database(db_name)
    res_col = db.get_collection(res_col_name)
    cfg_col = db.get_collection(cfg_col_name)

    """
    运行参数配置
    运行参数从数据库中读取
    """
    cfgs = list(cfg_col.find())
    cfg = cfgs[cfg_idx]
    logger.info("加载配置cfg：{}".format(cfg_idx))
    for key in cfg.keys():
        logger.info("{:<20}:   {}".format(key, cfg[key]))
    id = res_col.find({"statics": True})[0]["total_count"] + 1
    logger.info("开始编号{}实验".format(id))
    learn_type = cfg["learn_type"]
    dataset = cfg["dataset"]
    num_class = cfg["num_class"]
    model_name = cfg["model_name"]
    target_size = cfg["target_size"]
    batch_size = cfg["batch_size"]
    epoches = cfg["epoches"]

    keras_model_dir = cfg["keras_model_dir"]
    data_dir = cfg["data_dir"]
    if not os.path.exists(keras_model_dir):
        os.makedirs(keras_model_dir)
    tl_model_h5 = "{}_class{}_{}_tl_e{}_b{}.h5".format(dataset, num_class, model_name, epoches, batch_size)
    tl_model_h5 = os.path.join(keras_model_dir, tl_model_h5)
    if learn_type == "ft":
        GAP_LAYER = cfg["GAP_LAYER"]
        ft_model_h5 = "{}_class{}_{}_ft_e{}_b{}_gl{}.h5".format(dataset, num_class, model_name, epoches, batch_size,GAP_LAYER)
        ft_model_h5 = os.path.join(keras_model_dir, ft_model_h5)
        if "tl_model_h5" in options:  # ft学习时时可以指定fine tune的tl model
            tl_model_h5 = options["tl_model_h5"]

    """
    数据加载配置
    """
    train_dir = os.path.join(data_dir, "train")
    validate_dir = os.path.join(data_dir, "validate")
    test_dir = os.path.join(data_dir, "test")
    train_num = getNum(train_dir)
    validate_num = getNum(validate_dir)
    test_num = getNum(test_dir)
    logger.info("训练：{:10}，验证：{:10}，测试{:10}".format(train_num,validate_num,test_num))
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,# ((x/255)-0.5)*2  归一化到±1之间
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    train_generator = datagen.flow_from_directory(directory=train_dir,target_size=target_size,batch_size=batch_size)
    val_generator = datagen.flow_from_directory(directory=validate_dir,target_size=target_size,batch_size=batch_size)
    test_generator = datagen.flow_from_directory(directory=test_dir,target_size=target_size, batch_size=batch_size)

    """
    实验记录
    """
    res_col.insert_one({
        "id": id,
        "start_time": time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
        "learn_type": learn_type,
        "dataset": dataset,
        "num_class": num_class,
        "model_name": model_name,
        "target_size": target_size,
        "batch_size": batch_size,
        "epoches": epoches,
        "data_dir": data_dir,
        "sample_num": train_num+validate_num+test_num,
        "cfg_id": cfg["cfg_id"]
    })

    res_col.update({"statics": True}, {"$set": {"total_count": id}})

    if "tl" == learn_type:
        try:
            logger.info("进行transfer learning---------------------")
            if model_name == "xception":
                base_model = Xception(weights='imagenet', include_top=False)
            elif model_name == "inception_v3":
                base_model = InceptionV3(weights='imagenet', include_top=False)
            # 定制输出层
            x = base_model.output
            x = GlobalAveragePooling2D()(x)  # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(num_class, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)

            setup_to_transfer_learning(model, base_model)

            start_time = time.time()
            history_tl = model.fit_generator(generator=train_generator,
                                             steps_per_epoch=train_num // batch_size,
                                             epochs=epoches,
                                             validation_data=(x_val,y_val),
                                             # validation_data=val_generator,
                                             # validation_steps=validate_num // batch_size,
                                             class_weight='auto',
                                             callbacks=callbacks
                                             )
            logger.info(history_tl.history)
            logger.info("开始evaluate:")
            scores = model.evaluate_generator(test_generator, steps=test_num // batch_size)
            logger.info(scores)
            train_time = time.time()
            logger.info("训练时间：{}min".format(train_time - start_time) // 60)

            model.save(tl_model_h5)
            """
            记录成功运行结果
            """
            res_col.update({"id": id}, {"$set": {
                "GAP_LAYER": GAP_LAYER,
                "test_acc": scores[1],
                "train_time": (train_time-start_time) // 60,
                "tl_model_h5": tl_model_h5
            }})
        except Exception as e:
            """
            记录出错信息
            """
            logger.error(e)
            res_col.update({"id": id}, {"$set": {
                "err": str(e)
            }})

    elif "ft" == learn_type:
        try:
            logger.info("进行fine tuning learning---------------------")
            model = load_model(tl_model_h5)
            logger.info("{}模型已加载".format(tl_model_h5))
            logger.info("开始evaluate：")
            pre_scores = model.evaluate_generator(test_generator, steps=test_num // batch_size)
            logger.info(pre_scores)

            setup_to_fine_tune(model, GAP_LAYER)

            start_time = time.time()
            history_ft = model.fit_generator(generator=train_generator,
                                             steps_per_epoch=train_num // batch_size,
                                             epochs=epoches,
                                             validation_data=val_generator,
                                             validation_steps=validate_num // batch_size,
                                             class_weight='auto',
                                             callbacks=callbacks
                                             )
            logger.info(history_ft.history)
            logger.info("开始evaluate:")
            scores = model.evaluate_generator(test_generator, steps=test_num // batch_size)
            logger.info(scores)
            train_time = time.time()
            logger.info("训练时间：{}".format(train_time - start_time) // 60)
            model.save(ft_model_h5)
            res_col.update({"id": id}, {"$set": {
                "test_acc": scores[1],
                "pre_acc": pre_scores[1],
                "train_time": (train_time-start_time) // 60,
                "ft_model_h5": ft_model_h5
            }})
        except Exception as e:
            logger.error(e)
            res_col.update({"id": id}, {"$set": {
                "err": str(e)
            }})
    else:
        logger.error("未指定训练类型")


if __name__ == '__main__':
    # options = {
    #     "tl_model_h5":r"G:\keras_model\dpm_type_class10_xception_tl_e20_b16.h5"
    # }
    learn(-1)