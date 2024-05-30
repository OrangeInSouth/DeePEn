import json
import logging
import os
import queue
import sys
import time

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.instruction_generate import demon_prompt_generate, task_instruction_generate

import argparse
from src.main_model_thread import MainModelThread
from src.model_load import load_model
from src.assist_model_thread import AssistModelThread


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Process some files.')

    parser.add_argument('--config', help='the name of the file to process')
    parser.add_argument('--learning_rate', '-lr', default=0.0, type=float, required=False, help="learning_rate")
    parser.add_argument('--learning_epochs_nums', '-len', default=5, type=int, required=False,
                        help='learning_epochs_nums')
    parser.add_argument('--result_save_dir', '-rsd', default="./", type=str, required=False, help='result_save_dir')
    parser.add_argument('--run_mode', '-rm', default="dev", type=str, required=False, help='result_save_dir')
    parser.add_argument('--logits_processor_mode', '-lpm', default="based_on_probility_transfer_logits_local_processor",
                        type=str,
                        required=False,
                        help='logits_processor_mode')
    parser.add_argument('--device_compute', '-dp', default="cuda:0", type=str, required=False,
                        help='device_compute')
    parser.add_argument('--device0', '-d0', default="cuda:0", type=str, required=False,
                        help='device0')
    parser.add_argument('--device1', '-d1', default="cuda:1", type=str, required=False,
                        help='device1')
    parser.add_argument('--device2', '-d2', default="cuda:2", type=str, required=False,
                        help='device2')
    parser.add_argument('--device3', '-d3', default="cuda:3", type=str, required=False,
                        help='device3')
    parser.add_argument('--device4', '-d4', default="cuda:4", type=str, required=False,
                        help='device4')
    parser.add_argument('--device5', '-d5', default="cuda:5", type=str, required=False,
                        help='device5')
    parser.add_argument('--device6', '-d6', default="cuda:6", type=str, required=False,
                        help='device6')
    parser.add_argument('--device7', '-d7', default="cuda:7", type=str, required=False,
                        help='device7')
    parser.add_argument('--device8', '-d8', default="cuda:0", type=str, required=False,
                        help='device8')

    parser.add_argument('--ensemble_weight', '-ew',
                        nargs='+',
                        type=float,
                        default=[1.0], help='ensemble_weight', required=False
                        )

    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config_json = json.load(f)

    model_paths = config_json["model_path"]
    assist_model_count = len(model_paths) - 1

    main_model_path = config_json["model_path"]["main_model_path"]
    main_model_probability_transfer_matrix_path = config_json["probability_transfer_matrix_path"]["main_model_path"]
    main_model_system_template = config_json["prompt_template"]["main_model_system_template"]

    dev_file_path = config_json["file_path"]["dev_file_path"]
    test_file_path = config_json["file_path"]["test_file_path"]
    demon_file_path = config_json["file_path"]["demon_file_path"]

    instruction = config_json["prompt_template"]["instruction"]
    instruction_parameter = config_json["prompt_template"]["instruction_parameter"]
    max_new_tokens = config_json["run_parameter"]["max_new_tokens"]

    demon_parameter = config_json["prompt_template"]["demon_parameter"]
    result_process_parameter = config_json["result_process_parameter"]

    try:
        early_stop_string_list = result_process_parameter["early_stop_string_list"]
    except:
        early_stop_string_list = None

    result_save_dir = args.result_save_dir
    logits_processor_mode = args.logits_processor_mode
    if os.path.isdir(result_save_dir):
        pass
    else:
        os.makedirs(result_save_dir)

    learning_rate = args.learning_rate
    learning_epochs_nums = args.learning_epochs_nums
    run_mode = args.run_mode

    device_compute = args.device_compute

    device0 = args.device0
    device1 = args.device1
    device2 = args.device2
    device3 = args.device3
    device4 = args.device4
    device5 = args.device5
    device6 = args.device6
    device7 = args.device7
    device8 = args.device8
    device_list = [device0, device1, device2, device3, device4, device5, device6, device7, device8]
    ensemble_weight = args.ensemble_weight

    if len(model_paths) > 1:
        if ensemble_weight[0] != 1.0:
            assert len(ensemble_weight) == len(model_paths), "集成权重数和模型数必须相同"
            assert sum(ensemble_weight) == 1, "集成权重和须为1"
        else:
            ensemble_weight = [1.0 / len(model_paths)] * len(model_paths)

    input_file_path = dev_file_path if run_mode == "dev" else test_file_path

    logging.basicConfig(filename=os.path.join(result_save_dir,
                                              f'ensemble_lr{learning_rate}_learning_epochs_nums{learning_epochs_nums}.process.log'),
                        level=logging.DEBUG)
    logging.info(f'\n【config_json:】{config_json}')
    logging.info(f'\n【result_save_dir:】{result_save_dir}')
    logging.info(f'\n【learning_rate:】{learning_rate}')
    logging.info(f'\n【learning_epochs_nums:】{learning_epochs_nums}')

    main_model, main_model_tokenizer, main_model_streamer = load_model(main_model_path, "auto")
    assist_model_tokenizer_list = []
    main_model_probability_transfer_matrix_list = []
    assist_model_probability_transfer_matrix_list = []

    if assist_model_count != 0:
        main_model_probability_transfer_matrix = torch.load(main_model_probability_transfer_matrix_path,
                                                            map_location=device_compute)
        main_model_probability_transfer_matrix_list = [main_model_probability_transfer_matrix]

        assist_model_list = []
        assist_model_tokenizer_list = []
        assist_model_system_template_list = []
        assist_model_probability_transfer_matrix_list = []

        for index in range(1, assist_model_count + 1):
            assist_model, assist_model_tokenizer, _ = load_model(
                config_json["model_path"]["assist_model" + str(index) + "_path"], "auto")

            assist_model_list.append(assist_model)
            assist_model_tokenizer_list.append(assist_model_tokenizer)
            assist_model_system_template_list.append(
                config_json["prompt_template"]["assist_model" + str(index) + "_system_template"])
            assist_model_probability_transfer_matrix_list.append(
                torch.load(config_json["probability_transfer_matrix_path"]["assist_model" + str(index) + "_path"],
                           map_location=device_list[index]))

    # ================================================================
    result_file_path = os.path.join(result_save_dir,
                                    f'ensemble_lr{learning_rate}_learning_epochs_nums{learning_epochs_nums}.jsonl')
    try:
        with open(result_file_path, 'r') as file:
            lines = file.readlines()
            line_count = len(lines)
        start_index = line_count
    except:
        start_index = 0

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        try:
            demon_instruction, demon_count = demon_prompt_generate(demon_file_path, demon_parameter)
        except:
            demon_instruction = ""
            demon_count = 0
        contents = input_file.readlines()

        for index, line in enumerate(tqdm(contents[start_index:])):
            line = json.loads(line)

            task_instruction = task_instruction_generate(line, instruction_parameter)
            final_input_prompt = instruction + demon_instruction + task_instruction
            main_model_input = main_model_system_template.format(final_input_prompt)

            information_key_list = demon_parameter['key']
            information_dict = {}
            for key in information_key_list:
                information_dict[key] = line[key]
            information_dict['main_model_input'] = main_model_input
            information_dict['demon_count'] = demon_count
            information_dict['task_instruction'] = task_instruction
            information_dict['max_new_tokens'] = max_new_tokens
            information_dict['result_process_parameter'] = result_process_parameter
            information_dict['logits_processor_mode'] = logits_processor_mode
            information_dict['ensemble_weight'] = ensemble_weight

            ensemble_model_output_ids_queue = queue.Queue()

            assist_model_score_queue_list = []
            assist_model_input_list = []
            for index in range(0, assist_model_count):
                assist_model_score_queue_list.append(queue.Queue())
                assist_model_input_list.append(assist_model_system_template_list[index].format(final_input_prompt))

            main_model_thread = MainModelThread(main_model=main_model,
                                                main_model_tokenizer=main_model_tokenizer,
                                                assist_model_tokenizer=assist_model_tokenizer_list,
                                                information_dict=information_dict,
                                                learning_rate=learning_rate,
                                                learning_epochs_nums=learning_epochs_nums,
                                                result_save_dir=result_save_dir,
                                                ensemble_model_output_ids_queue=ensemble_model_output_ids_queue,
                                                assist_model_score_queue_list=assist_model_score_queue_list,
                                                main_model_probability_transfer_matrix_list=main_model_probability_transfer_matrix_list,
                                                assist_model_probability_transfer_matrix_list=assist_model_probability_transfer_matrix_list,
                                                device_compute=device_compute,
                                                device=device0,
                                                early_stop_string_list=early_stop_string_list
                                                )
            main_model_thread.start()

            for i in range(max_new_tokens):

                assist_model_thread_list = []
                for index in range(0, assist_model_count):
                    # print(assist_model_list[index])
                    assist_model_thread = AssistModelThread(model=assist_model_list[index],
                                                            model_tokenizer=assist_model_tokenizer_list[index],
                                                            assist_model_input=assist_model_input_list[index],
                                                            assist_model_score_queue=assist_model_score_queue_list[
                                                                index],
                                                            device=device_list[index],
                                                            result_save_dir=result_save_dir
                                                            )
                    assist_model_thread.start()
                    assist_model_thread_list.append(assist_model_thread)

                for assist_model_thread in assist_model_thread_list:
                    assist_model_thread.join()

                if max_new_tokens != 1:

                    try:
                        ensemble_model_generate_next_id = ensemble_model_output_ids_queue.get(block=True,
                                                                                              timeout=4 + 0.0167 * max_new_tokens).to(
                            device_compute)
                        logging.info(
                            f'{i}, {main_model_tokenizer.convert_ids_to_tokens(ensemble_model_generate_next_id)}')
                        print(i, main_model_tokenizer.convert_ids_to_tokens(ensemble_model_generate_next_id))
                    except:
                        break

                    temp_tensor = ensemble_model_generate_next_id
                    got_tokens = main_model_tokenizer.convert_ids_to_tokens(temp_tensor)
                    temp_tokens = got_tokens[:]

                    if isinstance(temp_tokens[0], bytes):
                        temp_tokens[0] = temp_tokens[0].decode("utf-8")
                    if temp_tokens[0].startswith('▁'):
                        new_token = " " + temp_tokens[0][1:]
                    else:
                        new_token = temp_tokens[0]
                    if new_token == "</s>":
                        break

                    for index in range(len(assist_model_input_list)):
                        assist_model_input_list[index] += "{}".format(new_token)

    time_elapsed = time.time() - start_time  # 获得时间差
    minutes = int(time_elapsed / 60)
    seconds = int(time_elapsed % 60)
    logging.info(f"\nTime taken: {minutes} min {seconds} sec")
    print('Time taken: {} min {} sec'.format(minutes, seconds))


if __name__ == '__main__':
    main()
