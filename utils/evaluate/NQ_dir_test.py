import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.evaluate.utils.NQ_evaluate_predictions import NQ_evaluate

import json
import os.path
import re


def result_write(result_path, sys_file_name, num_correct, num_total, accuracy):
    with open(os.path.join(result_path, 'EM_accuracy_all.jsonl'), 'a+', encoding='utf-8') as result_file:
        dict = {}

        match = re.search(r'lr(.*?)anchor_point_count(.*?)learning_epochs_nums(.*)', sys_file_name)
        lr, anchor_point_count, learning_epochs_nums = match.groups()
        dict['learning_rate'] = lr.strip('_')
        dict['accuracy'] = '{:.2f}'.format(accuracy * 100)
        dict['num_total'] = num_total
        dict['num_correct'] = num_correct
        dict['sys_file_path'] = os.path.join(result_path, sys_file_name)
        dict['anchor_point_count'] = anchor_point_count.strip('_')
        dict['learning_epochs_nums'] = learning_epochs_nums.strip('.jsonl')

        result_file.write(json.dumps(dict, ensure_ascii=False) + '\n')


def find_files_with_suffix(folder_path, suffix):
    # 使用os模块获取文件夹中所有文件的路径
    all_files = os.listdir(folder_path)
    # 筛选以指定后缀名结尾的文件
    filtered_files = [file for file in all_files if file.endswith(suffix)]
    return filtered_files


result_file_dir = sys.argv[1]

jsonl_files_list = find_files_with_suffix(result_file_dir, "5.jsonl")
jsonl_files_list.sort()
for sys_file_name in jsonl_files_list:
    file_path = os.path.join(result_file_dir, sys_file_name)
    num_correct, num_total, accuracy = NQ_evaluate(file_path, file_path)

    print('{:.2f}'.format(accuracy * 100))
    result_write(result_file_dir, sys_file_name, num_correct, num_total, accuracy)
