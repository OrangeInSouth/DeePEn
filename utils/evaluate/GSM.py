import json
import os
import os.path
import re
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def extract_last_num(text: str) -> str | None:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return str(num_str)
    else:
        return None


def is_correct(model_answer, answer):
    return model_answer == answer


def result_write(result_path, sys_file_name, num_correct, num_total, accuracy):
    with open(os.path.join(result_path, 'EM_accuracy.jsonl'), 'a+', encoding='utf-8') as result_file:
        dict = {}
        match = re.search(r'lr(.*?)anchor_point_count(.*?)learning_epochs_nums(.*)', sys_file_name)
        lr, anchor_point_count, learning_epochs_nums = match.groups()
        dict['learning_rate'] = lr.strip('_')
        dict['accuracy'] = '{:.2f}'.format(accuracy)
        dict['num_correct'] = num_correct
        dict['num_total'] = num_total
        dict['sys_file_path'] = os.path.join(result_path, sys_file_name)

        dict['learning_epochs_nums'] = learning_epochs_nums.strip('.jsonl')
        dict['anchor_point_count'] = anchor_point_count.strip('_')

        result_file.write(json.dumps(dict, ensure_ascii=False) + '\n')


def find_files_with_suffix(folder_path, suffix):
    # 使用os模块获取文件夹中所有文件的路径
    all_files = os.listdir(folder_path)
    # 筛选以指定后缀名结尾的文件
    filtered_files = [file for file in all_files if file.endswith(suffix)]
    return filtered_files


result_file_dir = sys.argv[1]

jsonl_files_list = find_files_with_suffix(result_file_dir, ".jsonl")
# print(pdf_files)
for jsonl_file in tqdm(jsonl_files_list):

    sys_file_name = jsonl_file
    print(jsonl_file)

    with open(os.path.join(result_file_dir, jsonl_file), 'r', encoding='utf-8') as f:
        contents = f.readlines()
        ref_file_dict = {}
        sys_file_dict = {}
        correct_count = 0
        for line in contents:
            json_obj = json.loads(line)
            json_obj['question'] = json_obj['question'].strip()

            json_obj['prediction'] = extract_last_num(json_obj['all'])
            # json_obj['prediction'] = extract_last_num(json_obj["fuse_generation"])
            # json_obj['prediction'] = extract_last_num(json_obj["best"])

            json_obj['answer'] = extract_last_num(json_obj['answer'])
            # print(json_obj['prediction'], json_obj['answer'])
            if is_correct(json_obj['prediction'], json_obj['answer']):
                correct_count += 1

    accuracy = correct_count / len(contents)
    num_correct = correct_count
    num_total = len(contents)
    print(num_correct)
    print(num_total)
    print('{:.2f}'.format(accuracy * 100))
    result_write(result_file_dir, sys_file_name, num_correct, num_total, accuracy * 100)
