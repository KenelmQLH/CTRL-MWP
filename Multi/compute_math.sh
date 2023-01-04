#!/bin/bash

pred_data_dir=$1
data_name=$2 # make
data_version=$3 # 97

cd preprocess

if [ -f ${pred_data_dir}/infer_test.jsonl ];then
    echo "[1] pred_ques fromated in CTRL already exists ... "
else
    echo "[1] creating pred_ques fromated in CTRL ... "
    python process_plan_data.py -data_dir ${pred_data_dir} -work_mode pred
fi

cd ..


echo "[2] compute equation_acc for gold_ques and pred_ques ... "
python test_plan.py -test_file ${pred_data_dir}/infer_test.jsonl -data_name $2 -data_version $3
