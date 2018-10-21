from pymongo import MongoClient,ASCENDING
import time

client = MongoClient()
db_name = "keras"
cfg_col_name = "experiment_cfg"
res_col_name = "experiment_res"
db = client.get_database(db_name)
res_col = db.get_collection(res_col_name)
cfg_col = db.get_collection(cfg_col_name)


"""
运行参数配置
"""
# id = time.time()
# pipelines = ["tl"]
# data_name = "dpm_type"
# num_class = 10
# model_name = "xception"
# target_size = (299, 299)
# batch_size = 16
# epoches = 10
# GAP_LAYER = 65
# keras_model_dir = "G:\\keras_model"

cfgs = list(cfg_col.find())
id = 0
for cfg in cfgs:
    if cfg["cfg_id"] > id:
        id = cfg["cfg_id"]
id = id + 1
cfg_col.insert_one({
    "cfg_id": id,
    "cfg_time": time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
    "learn_type": "tl",
    "dataset": "dpm_type",
    "num_class": 10,
    "model_name": "inception_v3",
    "target_size": (299, 299),
    "batch_size": 16,
    "epoches": 10,
    "keras_model_dir": "G:\\keras_model",
    "data_dir": "G:\\pic_data\\dpm_split_10",
    "comment": "dpm_n10_inceptionv3_e20_b16_tl_test_tensorborad"
})

cfgs = list(cfg_col.find())
print(cfgs[-1])
