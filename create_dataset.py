from PIL import Image
import os
import numpy
import random
import pickle


def create_array_dataset(data_name, src_dir, dst_dir, split=(6, 1, 1), shape=(256, 256), dtype="int8"):
    nb_class = len(os.listdir(src_dir))
    print(data_name, src_dir, dst_dir, split, shape)
    dst_file = "{}\\dataset_{}_nb_class_{}_split_{}_shape_{}_dtype_{}".format(
        dst_dir, data_name, nb_class, split[0], shape[0], dtype
    )

    dtypes = {
        "byte": numpy.byte,
        "short": numpy.short,
        "int": numpy.int,
        "int8": numpy.int8,
        "float": numpy.float,
        "float32": numpy.float32,
        "float64": numpy.float64
    }
    dtype = dtypes[dtype]

    class_id = 0
    classes = {}
    print(os.listdir(src_dir))


    train_lists = []
    val_lists = []
    test_lists = []

    for child in os.listdir(src_dir):

        sub_dir = os.path.join(src_dir, child)
        img_lists = os.listdir(sub_dir)
        unit = len(img_lists) // sum(split)
        random.shuffle(img_lists)

        class_id += 1
        classes[class_id] = child

        class_imgs = []
        print("precess dir : {}".format(sub_dir))
        i = 0
        for img in img_lists:
            try:
                img_array = Image.open(os.path.join(sub_dir, img))
                img_array = img_array.convert("RGB")
                class_imgs.append((img_array.resize(shape), class_id))
                i += 1
                if i % 10 == 0:
                    print("已经处理{}/{}".format(i, len(img_lists)))
            except Exception as e:
                print(e, img)

        train_lists.extend(class_imgs[: unit * split[0]])
        val_lists.extend(class_imgs[unit * split[0]: unit * (split[0] + split[1])])
        test_lists.extend(class_imgs[unit * (split[0] + split[1]):])

    x_train = numpy.zeros((len(train_lists), shape[0], shape[1], 3), dtype=dtype)
    y_train = numpy.zeros(len(train_lists))
    x_val = numpy.zeros((len(val_lists), shape[0], shape[1], 3), dtype=dtype)
    y_val = numpy.zeros(len(val_lists))
    x_test = numpy.zeros((len(test_lists), shape[0], shape[1], 3), dtype=dtype)
    y_test = numpy.zeros(len(test_lists))

    for i in range(len(train_lists)):
        x_train[i] = numpy.array(train_lists[i][0], dtype=numpy.float32)
        y_train[i] = train_lists[i][1]
    for i in range(len(val_lists)):
        x_val[i] = numpy.array(val_lists[i][0], dtype=numpy.float32)
        y_val[i] = val_lists[i][1]
    for i in range(len(test_lists)):
        x_test[i] = numpy.array(test_lists[i][0], dtype=numpy.float32)
        y_test[i] = test_lists[i][1]

    dataset = {
        "name": data_name,
        "classes": classes,
        "nb_class": nb_class,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test
    }
    if os.path.exists(dst_file):
        os.remove(dst_file)
    print("准备将数据集写入：{}".format(dst_file))
    with open(dst_file, 'wb') as f:
        pickle.dump(dataset, f)
        print("数据集写入成功：{}".format(dst_file))

    return dst_file


def create_rawpixel_dataset(basedir, f_dataset, batch_size, size):
    import pickle
    import os
    from PIL import Image
    import numpy as np
    # base_dir = r"G:\\pic_data\\dpm\\origin\\type10\\"
    x = np.zeros((5147, size, size, 3))
    y = np.zeros((5147,))
    i = 0
    for j, cls in enumerate(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, cls)
        for f_img in os.listdir(subdir):
            f_img = os.path.join(subdir, f_img)
            print(i, f_img)
            img = Image.open(f_img)
            img = img.resize((size, size))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_arr = np.asarray(img)
            x[i] = img_arr
            y[i] = j
            i += 1

    i = 0
    while i*batch_size <= len(x):
        with open("{}_batch_{}".format(f_dataset, i+1), 'wb') as f:
            pickle.dump(
                {
                "x": x[i*batch_size: (i+1)*batch_size],
                "y": y[i*batch_size: (i+1)*batch_size]
                },
                f
            )
            print("dump ", "{}_batch_{}".format(f_dataset, i+1))
            i += 1
    print("done")


if __name__ == '__main__':
    # src_dir = "G:\\pic_data\\dpm\\origin\\type10"
    # dst_dir = "G:\\pic_data\\dpm\\dataset\\array"
    # data_name = "dpm_type"
    # with open(r"G:\pic_data\dpm\dataset\array\dataset_dpm_type_nb_class_10_split_6_shape_299_dtype_int8",'rb') as all:
    #     dataset = pickle.load(all)
    #     val_dataset = {
    #         "x_val": dataset["x_val"],
    #         "y_val": dataset["y_val"]
    #     }
    #     with open(r"G:\pic_data\dpm\dataset\array\dataset_dpm_type_nb_class_10_split_6_shape_299_dtype_int8_val",'wb') as val_f:
    #         pickle.dump(val_dataset, val_f)

    base_dir = r"G:\\pic_data\\dpm\\origin\\type10\\"
    f_dataset = r"F:\jupyter-notebook\deeplearning\features\rawpixel_dpm_10_64"
    create_rawpixel_dataset(base_dir, f_dataset, 1000, 64)