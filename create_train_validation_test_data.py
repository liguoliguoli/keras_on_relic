import shutil
import os
import random

src_base_dir = "G:\\pic_data\\dpm"
target_base_dir = "G:\\pic_data\\dpm_split"

if os.path.exists(target_base_dir):
    shutil.rmtree(target_base_dir)
os.makedirs(target_base_dir)

for child in os.listdir(src_base_dir):
    print(child)
    sub_dir = os.path.join(src_base_dir, child)
    img_list = os.listdir(sub_dir)
    random.shuffle(img_list)
    unit = len(img_list) // 5
    print(unit)

    train_dir = os.path.join(target_base_dir, 'train', child)
    validate_dir = os.path.join(target_base_dir, 'validate', child)
    test_dir = os.path.join(target_base_dir, 'test', child)
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(validate_dir):
        shutil.rmtree(validate_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(train_dir)
    os.makedirs(validate_dir)
    os.makedirs(test_dir)

    for train_img in img_list[: unit * 3]:
        src_img = os.path.join(sub_dir, train_img)
        print(src_img)
        dst_img = os.path.join(train_dir, train_img)
        print(dst_img)
        shutil.copy2(src_img, dst_img)
    for validate_img in img_list[unit * 3: unit * 4]:
        src_img = os.path.join(sub_dir, validate_img)
        print(src_img)
        dst_img = os.path.join(validate_dir, validate_img)
        print(dst_img)
        shutil.copy2(src_img, dst_img)
    for test_img in img_list[unit * 4:]:
        src_img = os.path.join(sub_dir, test_img)
        print(src_img)
        dst_img = os.path.join(test_dir, test_img)
        print(dst_img)
        shutil.copy2(src_img, dst_img)
