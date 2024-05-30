import json
import os
import threading

from transformers import LogitsProcessorList, TextStreamer

from src.logits_processor.model_processor_factory import ModelProcessorFactory
from utils.answer_extract import answer_extract


class MainModelThread(threading.Thread):
    def __init__(self, main_model, main_model_tokenizer, assist_model_tokenizer, information_dict,
                 learning_rate, learning_epochs_nums, result_save_dir,
                 ensemble_model_output_ids_queue,
                 assist_model_score_queue_list, main_model_probability_transfer_matrix_list,
                 assist_model_probability_transfer_matrix_list, device, device_compute, early_stop_string_list=None):
        self.model = main_model
        self.tokenizer = main_model_tokenizer
        self.assist_model_tokenizer = assist_model_tokenizer
        self.information_dict = information_dict
        self.model_streamer = TextStreamer(self.tokenizer)
        self.learning_rate = learning_rate
        self.learning_epochs_nums = learning_epochs_nums
        self.result_save_dir = result_save_dir
        self.ensemble_model_output_ids_queue = ensemble_model_output_ids_queue
        self.assist_model_score_queue_list = assist_model_score_queue_list
        self.main_model_probability_transfer_matrix_list = main_model_probability_transfer_matrix_list
        self.assist_model_probability_transfer_matrix_list = assist_model_probability_transfer_matrix_list
        self.device = device
        self.device_compute = device_compute
        self.early_stop_string_list = early_stop_string_list

        super().__init__()

    def run(self) -> None:
        main_model_logits_processor_list = LogitsProcessorList()

        processor_factory = ModelProcessorFactory()

        # 传递其他参数
        additional_kwargs = {
            "learning_rate": self.learning_rate,
            "ensemble_weight": self.information_dict['ensemble_weight'],
            "learning_epochs_nums": self.learning_epochs_nums,
            "ensemble_model_output_ids_queue": self.ensemble_model_output_ids_queue,
            "assist_model_score_queue_list": self.assist_model_score_queue_list,

            "main_model_probability_transfer_matrix_list": self.main_model_probability_transfer_matrix_list,
            "assist_model_probability_transfer_matrix_list": self.assist_model_probability_transfer_matrix_list,
            "result_save_dir": self.result_save_dir,
            "main_model_tokenizer": self.tokenizer,
            "assist_model_tokenizer": self.assist_model_tokenizer,
            "device": self.device,
            "device_compute": self.device_compute,
            "early_stop_string_list": self.early_stop_string_list,
        }
        # 创建对象
        logits_processor_mode = self.information_dict['logits_processor_mode']
        logits_processor_instance = processor_factory.create_processor(logits_processor_mode,
                                                                       **additional_kwargs)
        main_model_logits_processor_list.append(logits_processor_instance)
        # main_model_logits_processor_list.append()

        main_model_input = self.information_dict['main_model_input']
        max_new_tokens = self.information_dict['max_new_tokens']
        main_model_input_ids = self.tokenizer(main_model_input, return_tensors="pt",
                                              add_special_tokens=False).input_ids.to(self.device)
        generation_kwargs = {
            "input_ids": main_model_input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            # "pad_token_id": self.tokenizer.pad_token_id
        }

        # generate_ids = self.model.generate(**generation_kwargs,pad_token_id=self.tokenizer.eos_token_id,
        #                                    logits_processor=main_model_logits_processor_list,
        #                                    streamer=self.model_streamer)
        generate_ids = self.model.generate(**generation_kwargs, pad_token_id=self.tokenizer.eos_token_id,
                                           logits_processor=main_model_logits_processor_list)

        text = self.tokenizer.decode(generate_ids[0])
        # print(text)
        result_process_parameter = self.information_dict['result_process_parameter']
        split_key_before_list = result_process_parameter["split_key_before"]
        split_key_behind_list = result_process_parameter["split_key_behind"]

        model_answer, prediction = answer_extract(text, self.information_dict['demon_count'], split_key_before_list,
                                                  split_key_behind_list)
        print(self.information_dict['question'])
        print(prediction.strip())
        model_answer_dict = {'answer': self.information_dict['answer'],
                             'prediction': prediction.strip(), 'main_model_input': main_model_input, 'all': text,
                             'model_answer': model_answer,
                             'question': self.information_dict['question']}

        result_file_path = os.path.join(self.result_save_dir,
                                        f'ensemble_lr{self.learning_rate}_learning_epochs_nums{self.learning_epochs_nums}.jsonl')
        with open(result_file_path, 'a+', encoding='utf-8') as result_file:
            result_file.write(json.dumps(model_answer_dict, ensure_ascii=False) + '\n')
