import json
import argparse
from EasyData.FileHandler import check2mkdir, path_append, abs_current_dir
import os
import sys
CUR_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def add_args(parser):
    parser.add_argument('-data_name', type=str, default="hmwp")
    parser.add_argument('-data_version', type=int, default=97)


parser = argparse.ArgumentParser(description='[Get args for wrok data]')
add_args(parser)
work_opt = parser.parse_args()


DATA_NAME = work_opt.data_name
# 读取 题目数据
DATA_DICT = {
    "make": ["chinese", "muti_equation"],
    "hmwp": ["chinese", "muti_equation"],
    "dophin": ["english", "muti_equation"],
    "equation": ["chinese", "muti_equation"],

    "arithmetic": ["english", "linear_expression"],
    "mawps": ["english", "linear_expression"],
    "math23k_en": ["english", "linear_expression"],
    "linear": ["english", "linear_expression"],
    "nl4opt": ["english", "optimization"],
}
DATA_TYPE = DATA_DICT[DATA_NAME][1]
DATA_VERSION = work_opt.data_version



filename_train = open(path_append(abs_current_dir(__file__), f'../data/{DATA_NAME}/dev_test/{DATA_VERSION}/{DATA_NAME}_train.jsonl', to_str=True), 'r', encoding="utf-8")
filename_dev = open(path_append(abs_current_dir(__file__), f'../data/{DATA_NAME}/dev_test/{DATA_VERSION}/{DATA_NAME}_dev.jsonl', to_str=True), 'r', encoding="utf-8")
filename_test = open(path_append(abs_current_dir(__file__), f'../data/{DATA_NAME}/dev_test/{DATA_VERSION}/{DATA_NAME}_test.jsonl', to_str=True), 'r', encoding="utf-8")
data = []
for line in filename_train:
    temp = json.loads(line)
    data.append(temp)
for line in filename_dev:
    temp = json.loads(line)
    data.append(temp)
for line in filename_test:
    temp = json.loads(line)
    data.append(temp)

fold_size = int(len(data) * 0.2)
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(data[fold_start:fold_end])
fold_pairs.append(data[(fold_size * 4):])

for fold in range(5):
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]
    # 5-fold ==> mtokens
    check2mkdir(path_append(abs_current_dir(__file__), f'../data/{DATA_NAME}/mtokens/{DATA_VERSION}/{DATA_NAME}_fold'+str(fold)+'_train.jsonl', to_str=True))
    f = open(path_append(abs_current_dir(__file__), f'../data/{DATA_NAME}/mtokens/{DATA_VERSION}/{DATA_NAME}_fold'+str(fold)+'_train.jsonl', to_str=True), 'w', encoding="utf-8")
    for d in pairs_trained:
        json.dump(d, f)
        f.write('\n')
    f.close()
    f = open(path_append(abs_current_dir(__file__), f'../data/{DATA_NAME}/mtokens/{DATA_VERSION}/{DATA_NAME}_fold'+str(fold)+'_test.jsonl', to_str=True), 'w', encoding="utf-8")
    for d in pairs_tested:
        json.dump(d, f)
        f.write('\n')
    f.close()


# filename_train = open('../data/hmwp/dev_test/HMWP_train.jsonl', 'r')
# filename_dev = open('../data/hmwp/dev_test/HMWP_dev.jsonl', 'r')
# filename_test = open('../data/hmwp/dev_test/HMWP_test.jsonl', 'r')
# data = []
# for line in filename_train:
#     temp = json.loads(line)
#     data.append(temp)
# for line in filename_dev:
#     temp = json.loads(line)
#     data.append(temp)
# for line in filename_test:
#     temp = json.loads(line)
#     data.append(temp)

# fold_size = int(len(data) * 0.2)
# fold_pairs = []
# for split_fold in range(4):
#     fold_start = fold_size * split_fold
#     fold_end = fold_size * (split_fold + 1)
#     fold_pairs.append(data[fold_start:fold_end])
# fold_pairs.append(data[(fold_size * 4):])

# for fold in range(5):
#     pairs_tested = []
#     pairs_trained = []
#     for fold_t in range(5):
#         if fold_t == fold:
#             pairs_tested += fold_pairs[fold_t]
#         else:
#             pairs_trained += fold_pairs[fold_t]
#     f = open('../data/hmwp/mtokens/HMWP_fold'+str(fold)+'_train.jsonl', 'w')
#     for d in pairs_trained:
#         json.dump(d, f, ensure_ascii=False)
#         f.write('\n')
#     f.close()
#     f = open('../data/hmwp/mtokens/HMWP_fold'+str(fold)+'_test.jsonl', 'w')
#     for d in pairs_tested:
#         json.dump(d, f, ensure_ascii=False)
#         f.write('\n')
#     f.close()