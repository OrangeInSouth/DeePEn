import json
import math
import os
import queue

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from transformers import LogitsProcessor


class InternLMPPLBasedOnProbabilityTransferLogitsProcessor(LogitsProcessor):
    def __init__(self, learning_rate, learning_epochs_nums, ensemble_weight,
                 ensemble_model_output_ids_queue, assist_model_score_queue_list,
                 main_model_probability_transfer_matrix_list,
                 assist_model_probability_transfer_matrix_list, result_save_dir, main_model_tokenizer,
                 assist_model_tokenizer, device, device_compute, early_stop_string_list=None):
        self.learning_rate = learning_rate
        self.ensemble_weight = ensemble_weight
        self.assist_model_score_queue_list = assist_model_score_queue_list
        self.learning_epochs_nums = learning_epochs_nums
        self.ensemble_model_output_ids_queue = ensemble_model_output_ids_queue
        self.main_model_probability_transfer_matrix_list = main_model_probability_transfer_matrix_list
        self.assist_model_probability_transfer_matrix_list = assist_model_probability_transfer_matrix_list
        self.result_save_dir = result_save_dir
        self.main_model_tokenizer = main_model_tokenizer
        self.assist_model_tokenizer_list = assist_model_tokenizer
        self.device = device
        self.device_compute = device_compute
        self.early_stop_string_list = early_stop_string_list

    def calculate_ppl(self, logits, labels):
        neg_log_likelihood = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
        ppl = torch.exp(neg_log_likelihood)
        return ppl.item()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        ensemble_process_file_path = os.path.join(self.result_save_dir,
                                                  f'ensemble_lr{self.learning_rate}_anchor_point_count_all_learning_epochs_nums_5.log')
        json_object = {}
        main_model_only_flag = False
        assist_model_generate_ids_logits_list = []
        for index, queue_instance in enumerate(self.assist_model_score_queue_list):
            try:
                value = queue_instance.get(block=True, timeout=5)
                assist_model_generate_ids_logits_list.append(value)

            except queue.Empty:
                print(f"aux model{index}【not received】\n")
                assist_model_generate_ids_logits_list.append(None)
                main_model_only_flag = True

        if math.fabs(self.learning_rate) <= 1e-6:
            main_model_only_flag = True

        if len(assist_model_generate_ids_logits_list) == 0:
            main_model_only_flag = True
        # ▁A ▁B
        A_index = 493
        B_index = 556
        C_index = 487
        D_index = 553
        token_ABCD_index_list = [A_index, B_index, C_index, D_index]

        ppl_ABCD_list = []
        optionA = torch.zeros_like(scores).to(self.device)
        optionA[:, A_index] = 1
        optionB = torch.zeros_like(scores).to(self.device)
        optionB[:, B_index] = 1
        optionC = torch.zeros_like(scores).to(self.device)
        optionC[:, C_index] = 1
        optionD = torch.zeros_like(scores).to(self.device)
        optionD[:, D_index] = 1

        if not main_model_only_flag:
            main_model_generate_ids_logits = Variable(scores, requires_grad=True).to(torch.float32).to(
                self.main_model_probability_transfer_matrix_list[0].device)

            with torch.no_grad():
                main_model_generate_ids_probs = nn.functional.softmax(main_model_generate_ids_logits, dim=-1).float()

                main_model_generate_ids_probs_values, main_model_generate_ids_probs_indices = torch.topk(
                    main_model_generate_ids_probs, k=10)
                json_object[f'origin_main_top_tokens'] = self.main_model_tokenizer.convert_ids_to_tokens(
                    main_model_generate_ids_probs_indices.tolist()[0])

                main_model_relative_representation_probs = torch.mm(main_model_generate_ids_probs,
                                                                    self.main_model_probability_transfer_matrix_list[
                                                                        0]).to(self.device_compute)
                main_model_relative_values, main_model_relative_indices = torch.topk(
                    main_model_relative_representation_probs, k=10)
                json_object[f'main_rel_values'] = main_model_relative_values.tolist()[0]
                json_object[f'main_rel_indices'] = main_model_relative_indices.tolist()[0]

                model_relative_representation_probs_list = [main_model_relative_representation_probs]

                for index, (assist_model_generate_ids_logits, assist_model_probability_transfer_matrix) in enumerate(
                        zip(
                                assist_model_generate_ids_logits_list,
                                self.assist_model_probability_transfer_matrix_list)):
                    assist_model_generate_ids_probs = nn.functional.softmax(assist_model_generate_ids_logits,
                                                                            dim=-1).float().to(
                        assist_model_probability_transfer_matrix.device)

                    values, indices = torch.topk(assist_model_generate_ids_probs, k=10)
                    json_object[f'origin_aux_{index}_top_tokens'] = self.assist_model_tokenizer_list[
                        index].convert_ids_to_tokens(indices.tolist()[0])

                    assist_model_relative_representation_probs = torch.mm(assist_model_generate_ids_probs,
                                                                          assist_model_probability_transfer_matrix).to(
                        self.device_compute)

                    assist_model_relative_values, assist_model_relative_indices = torch.topk(
                        assist_model_relative_representation_probs, k=10)
                    json_object[f'aux_rel_values_{index}'] = assist_model_relative_values.tolist()[0]
                    json_object[f'aux_rel_indices_{index}'] = assist_model_relative_indices.tolist()[0]

                    model_relative_representation_probs_list.append(assist_model_relative_representation_probs)

            json_object[f'ensemble_weight'] = self.ensemble_weight

            average_probs = torch.zeros_like(main_model_relative_representation_probs)
            # print(self.ensemble_weight)
            for weight, probs in zip(self.ensemble_weight, model_relative_representation_probs_list):
                average_probs += weight * probs

            average_relative_probs_values, average_relative_probs_indices = torch.topk(
                average_probs, k=10)

            json_object[f'average_rel_probs_values'] = average_relative_probs_values.tolist()[0]
            json_object[f'average_rel_probs_indices'] = average_relative_probs_indices.tolist()[0]

            torch.set_grad_enabled(True)

            main_model_generate_ids_logits = main_model_generate_ids_logits.to(self.device_compute).detach().clone().to(
                torch.float32)
            main_model_generate_ids_logits.requires_grad_(True)

            local_learning_rate = self.learning_rate
            criterion = nn.KLDivLoss()

            optimizer = torch.optim.AdamW(params=[main_model_generate_ids_logits],
                                          lr=local_learning_rate,
                                          betas=(0.9, 0.999))

            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=local_learning_rate / 4)

            for i in range(1, self.learning_epochs_nums + 1):
                main_model_generate_ids_probs = nn.functional.softmax(main_model_generate_ids_logits, dim=-1).float()
                main_model_relative_representation_probs = torch.mm(main_model_generate_ids_probs,
                                                                    self.main_model_probability_transfer_matrix_list[0])

                log_main_probs = torch.log(main_model_relative_representation_probs)
                loss = criterion(log_main_probs, average_probs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                main_model_generate_ids_logits_probs_values, main_model_generate_ids_logits_indices = torch.topk(
                    torch.nn.functional.softmax(main_model_generate_ids_logits, dim=-1), k=10)
                json_object[f'main_model_generate_ids_logits_probs_values_{i}'] = \
                    main_model_generate_ids_logits_probs_values.tolist()[0]
                json_object[f'main_model_generate_ids_logits_indices_{i}'] = \
                    self.main_model_tokenizer.convert_ids_to_tokens(
                        main_model_generate_ids_logits_indices.tolist()[0])

            torch.set_grad_enabled(False)
            optionA_ppl = self.calculate_ppl(main_model_generate_ids_logits.to(self.device), optionA)
            optionB_ppl = self.calculate_ppl(main_model_generate_ids_logits.to(self.device), optionB)
            optionC_ppl = self.calculate_ppl(main_model_generate_ids_logits.to(self.device), optionC)
            optionD_ppl = self.calculate_ppl(main_model_generate_ids_logits.to(self.device), optionD)

            ppl_ABCD_list.append(optionA_ppl)
            ppl_ABCD_list.append(optionB_ppl)
            ppl_ABCD_list.append(optionC_ppl)
            ppl_ABCD_list.append(optionD_ppl)
            next_tokens_id = token_ABCD_index_list[ppl_ABCD_list.index(min(ppl_ABCD_list))]

            main_model_generate_ids_logits_probs_values, main_model_generate_ids_logits_indices = torch.topk(
                torch.nn.functional.softmax(main_model_generate_ids_logits, dim=-1), k=10)
            json_object[f'main_model_generate_ids_logits_probs_values_final'] = \
                main_model_generate_ids_logits_probs_values.tolist()[0]
            json_object[f'main_model_generate_ids_logits_indices_final'] = \
                self.main_model_tokenizer.convert_ids_to_tokens(
                    main_model_generate_ids_logits_indices.tolist()[0])

            with open(ensemble_process_file_path, "a+", encoding="utf-8") as process_file:
                process_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')

            output = torch.zeros_like(scores).to(self.device)
            output[:, next_tokens_id] = float('inf')
            return output
        else:
            optionA_ppl = self.calculate_ppl(scores, optionA)
            optionB_ppl = self.calculate_ppl(scores, optionB)
            optionC_ppl = self.calculate_ppl(scores, optionC)
            optionD_ppl = self.calculate_ppl(scores, optionD)
            ppl_ABCD_list.append(optionA_ppl)
            ppl_ABCD_list.append(optionB_ppl)
            ppl_ABCD_list.append(optionC_ppl)
            ppl_ABCD_list.append(optionD_ppl)
            next_tokens_id = token_ABCD_index_list[ppl_ABCD_list.index(min(ppl_ABCD_list))]
            output = torch.zeros_like(scores).to(self.device)
            output[:, next_tokens_id] = float('inf')
            return output


class YiPPLBasedOnProbabilityTransferLogitsProcessor(LogitsProcessor):
    def __init__(self, learning_rate, learning_epochs_nums, ensemble_weight,
                 ensemble_model_output_ids_queue, assist_model_score_queue_list,
                 main_model_probability_transfer_matrix_list,
                 assist_model_probability_transfer_matrix_list, result_save_dir, main_model_tokenizer,
                 assist_model_tokenizer, device, device_compute, early_stop_string_list=None):
        self.learning_rate = learning_rate
        self.ensemble_weight = ensemble_weight
        self.assist_model_score_queue_list = assist_model_score_queue_list
        self.learning_epochs_nums = learning_epochs_nums
        self.ensemble_model_output_ids_queue = ensemble_model_output_ids_queue
        self.main_model_probability_transfer_matrix_list = main_model_probability_transfer_matrix_list
        self.assist_model_probability_transfer_matrix_list = assist_model_probability_transfer_matrix_list
        self.result_save_dir = result_save_dir
        self.main_model_tokenizer = main_model_tokenizer
        self.assist_model_tokenizer_list = assist_model_tokenizer
        self.device = device
        self.device_compute = device_compute
        self.early_stop_string_list = early_stop_string_list

    def calculate_ppl(self, logits, labels):
        neg_log_likelihood = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
        ppl = torch.exp(neg_log_likelihood)
        return ppl.item()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        ensemble_process_file_path = os.path.join(self.result_save_dir,
                                                  f'ensemble_lr{self.learning_rate}_anchor_point_count_all_learning_epochs_nums_5.log')
        json_object = {}
        main_model_only_flag = False
        assist_model_generate_ids_logits_list = []
        for index, queue_instance in enumerate(self.assist_model_score_queue_list):
            try:
                value = queue_instance.get(block=True, timeout=5)
                assist_model_generate_ids_logits_list.append(value)

            except queue.Empty:
                print(f"aux model{index}【not received】\n")
                assist_model_generate_ids_logits_list.append(None)
                main_model_only_flag = True

        if math.fabs(self.learning_rate) <= 1e-6:
            main_model_only_flag = True

        if len(assist_model_generate_ids_logits_list) == 0:
            main_model_only_flag = True
        # ▁A ▁B
        A_index = 647
        B_index = 690
        C_index = 650
        D_index = 723
        token_ABCD_index_list = [A_index, B_index, C_index, D_index]

        ppl_ABCD_list = []
        optionA = torch.zeros_like(scores).to(self.device)
        optionA[:, A_index] = 1
        optionB = torch.zeros_like(scores).to(self.device)
        optionB[:, B_index] = 1
        optionC = torch.zeros_like(scores).to(self.device)
        optionC[:, C_index] = 1
        optionD = torch.zeros_like(scores).to(self.device)
        optionD[:, D_index] = 1

        if not main_model_only_flag:
            main_model_generate_ids_logits = Variable(scores, requires_grad=True).to(torch.float32).to(
                self.main_model_probability_transfer_matrix_list[0].device)

            with torch.no_grad():
                main_model_generate_ids_probs = nn.functional.softmax(main_model_generate_ids_logits, dim=-1).float()

                main_model_generate_ids_probs_values, main_model_generate_ids_probs_indices = torch.topk(
                    main_model_generate_ids_probs, k=10)
                json_object[f'origin_main_top_tokens'] = self.main_model_tokenizer.convert_ids_to_tokens(
                    main_model_generate_ids_probs_indices.tolist()[0])

                main_model_relative_representation_probs = torch.mm(main_model_generate_ids_probs,
                                                                    self.main_model_probability_transfer_matrix_list[
                                                                        0]).to(self.device_compute)
                main_model_relative_values, main_model_relative_indices = torch.topk(
                    main_model_relative_representation_probs, k=10)
                json_object[f'main_rel_values'] = main_model_relative_values.tolist()[0]
                json_object[f'main_rel_indices'] = main_model_relative_indices.tolist()[0]

                model_relative_representation_probs_list = [main_model_relative_representation_probs]

                for index, (assist_model_generate_ids_logits, assist_model_probability_transfer_matrix) in enumerate(
                        zip(
                                assist_model_generate_ids_logits_list,
                                self.assist_model_probability_transfer_matrix_list)):
                    assist_model_generate_ids_probs = nn.functional.softmax(assist_model_generate_ids_logits,
                                                                            dim=-1).float().to(
                        assist_model_probability_transfer_matrix.device)

                    values, indices = torch.topk(assist_model_generate_ids_probs, k=10)
                    json_object[f'origin_aux_{index}_top_tokens'] = self.assist_model_tokenizer_list[
                        index].convert_ids_to_tokens(indices.tolist()[0])

                    assist_model_relative_representation_probs = torch.mm(assist_model_generate_ids_probs,
                                                                          assist_model_probability_transfer_matrix).to(
                        self.device_compute)

                    assist_model_relative_values, assist_model_relative_indices = torch.topk(
                        assist_model_relative_representation_probs, k=10)
                    json_object[f'aux_rel_values_{index}'] = assist_model_relative_values.tolist()[0]
                    json_object[f'aux_rel_indices_{index}'] = assist_model_relative_indices.tolist()[0]

                    model_relative_representation_probs_list.append(assist_model_relative_representation_probs)

            json_object[f'ensemble_weight'] = self.ensemble_weight

            average_probs = torch.zeros_like(main_model_relative_representation_probs)
            # print(self.ensemble_weight)
            for weight, probs in zip(self.ensemble_weight, model_relative_representation_probs_list):
                average_probs += weight * probs

            average_relative_probs_values, average_relative_probs_indices = torch.topk(
                average_probs, k=10)

            json_object[f'average_rel_probs_values'] = average_relative_probs_values.tolist()[0]
            json_object[f'average_rel_probs_indices'] = average_relative_probs_indices.tolist()[0]

            torch.set_grad_enabled(True)

            main_model_generate_ids_logits = main_model_generate_ids_logits.to(self.device_compute).detach().clone().to(
                torch.float32)
            main_model_generate_ids_logits.requires_grad_(True)

            local_learning_rate = self.learning_rate
            criterion = nn.KLDivLoss()

            optimizer = torch.optim.AdamW(params=[main_model_generate_ids_logits],
                                          lr=local_learning_rate,
                                          betas=(0.9, 0.999))

            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=local_learning_rate / 4)

            for i in range(1, self.learning_epochs_nums + 1):
                main_model_generate_ids_probs = nn.functional.softmax(main_model_generate_ids_logits, dim=-1).float()
                main_model_relative_representation_probs = torch.mm(main_model_generate_ids_probs,
                                                                    self.main_model_probability_transfer_matrix_list[0])

                log_main_probs = torch.log(main_model_relative_representation_probs)
                loss = criterion(log_main_probs, average_probs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                main_model_generate_ids_logits_probs_values, main_model_generate_ids_logits_indices = torch.topk(
                    torch.nn.functional.softmax(main_model_generate_ids_logits, dim=-1), k=10)
                json_object[f'main_model_generate_ids_logits_probs_values_{i}'] = \
                    main_model_generate_ids_logits_probs_values.tolist()[0]
                json_object[f'main_model_generate_ids_logits_indices_{i}'] = \
                    self.main_model_tokenizer.convert_ids_to_tokens(
                        main_model_generate_ids_logits_indices.tolist()[0])

            torch.set_grad_enabled(False)
            optionA_ppl = self.calculate_ppl(main_model_generate_ids_logits.to(self.device), optionA)
            optionB_ppl = self.calculate_ppl(main_model_generate_ids_logits.to(self.device), optionB)
            optionC_ppl = self.calculate_ppl(main_model_generate_ids_logits.to(self.device), optionC)
            optionD_ppl = self.calculate_ppl(main_model_generate_ids_logits.to(self.device), optionD)

            ppl_ABCD_list.append(optionA_ppl)
            ppl_ABCD_list.append(optionB_ppl)
            ppl_ABCD_list.append(optionC_ppl)
            ppl_ABCD_list.append(optionD_ppl)
            next_tokens_id = token_ABCD_index_list[ppl_ABCD_list.index(min(ppl_ABCD_list))]

            main_model_generate_ids_logits_probs_values, main_model_generate_ids_logits_indices = torch.topk(
                torch.nn.functional.softmax(main_model_generate_ids_logits, dim=-1), k=10)
            json_object[f'main_model_generate_ids_logits_probs_values_final'] = \
                main_model_generate_ids_logits_probs_values.tolist()[0]
            json_object[f'main_model_generate_ids_logits_indices_final'] = \
                self.main_model_tokenizer.convert_ids_to_tokens(
                    main_model_generate_ids_logits_indices.tolist()[0])

            with open(ensemble_process_file_path, "a+", encoding="utf-8") as process_file:
                process_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')

            output = torch.zeros_like(scores).to(self.device)
            output[:, next_tokens_id] = float('inf')
            return output
        else:
            optionA_ppl = self.calculate_ppl(scores, optionA)
            optionB_ppl = self.calculate_ppl(scores, optionB)
            optionC_ppl = self.calculate_ppl(scores, optionC)
            optionD_ppl = self.calculate_ppl(scores, optionD)
            ppl_ABCD_list.append(optionA_ppl)
            ppl_ABCD_list.append(optionB_ppl)
            ppl_ABCD_list.append(optionC_ppl)
            ppl_ABCD_list.append(optionD_ppl)
            next_tokens_id = token_ABCD_index_list[ppl_ABCD_list.index(min(ppl_ABCD_list))]
            output = torch.zeros_like(scores).to(self.device)
            output[:, next_tokens_id] = float('inf')
            return output


class YiPPLBasedOnProbabilityTransferLogitsPIQAProcessor(LogitsProcessor):
    def __init__(self, learning_rate, learning_epochs_nums, ensemble_weight,
                 ensemble_model_output_ids_queue, assist_model_score_queue_list,
                 main_model_probability_transfer_matrix_list,
                 assist_model_probability_transfer_matrix_list, result_save_dir, main_model_tokenizer,
                 assist_model_tokenizer, device, device_compute, early_stop_string_list=None):
        self.learning_rate = learning_rate
        self.ensemble_weight = ensemble_weight
        self.assist_model_score_queue_list = assist_model_score_queue_list
        self.learning_epochs_nums = learning_epochs_nums
        self.ensemble_model_output_ids_queue = ensemble_model_output_ids_queue
        self.main_model_probability_transfer_matrix_list = main_model_probability_transfer_matrix_list
        self.assist_model_probability_transfer_matrix_list = assist_model_probability_transfer_matrix_list
        self.result_save_dir = result_save_dir
        self.main_model_tokenizer = main_model_tokenizer
        self.assist_model_tokenizer_list = assist_model_tokenizer
        self.device = device
        self.device_compute = device_compute
        self.early_stop_string_list = early_stop_string_list

    def calculate_ppl(self, logits, labels):
        neg_log_likelihood = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
        ppl = torch.exp(neg_log_likelihood)
        return ppl.item()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        ensemble_process_file_path = os.path.join(self.result_save_dir,
                                                  f'ensemble_lr{self.learning_rate}_anchor_point_count_all_learning_epochs_nums_5.log')
        json_object = {}
        main_model_only_flag = False
        assist_model_generate_ids_logits_list = []
        for index, queue_instance in enumerate(self.assist_model_score_queue_list):
            try:
                value = queue_instance.get(block=True, timeout=5)
                assist_model_generate_ids_logits_list.append(value)

            except queue.Empty:
                print(f"aux model{index}【not received】\n")
                assist_model_generate_ids_logits_list.append(None)
                main_model_only_flag = True

        if math.fabs(self.learning_rate) <= 1e-6:
            main_model_only_flag = True

        if len(assist_model_generate_ids_logits_list) == 0:
            main_model_only_flag = True
        # ▁A ▁B
        A_index = 647
        B_index = 690
        # C_index = 650
        # D_index = 723
        token_ABCD_index_list = [A_index, B_index]

        ppl_ABCD_list = []
        optionA = torch.zeros_like(scores).to(self.device)
        optionA[:, A_index] = 1
        optionB = torch.zeros_like(scores).to(self.device)
        optionB[:, B_index] = 1

        if not main_model_only_flag:
            main_model_generate_ids_logits = Variable(scores, requires_grad=True).to(torch.float32).to(
                self.main_model_probability_transfer_matrix_list[0].device)

            with torch.no_grad():
                main_model_generate_ids_probs = nn.functional.softmax(main_model_generate_ids_logits, dim=-1).float()

                main_model_generate_ids_probs_values, main_model_generate_ids_probs_indices = torch.topk(
                    main_model_generate_ids_probs, k=10)
                json_object[f'origin_main_top_tokens'] = self.main_model_tokenizer.convert_ids_to_tokens(
                    main_model_generate_ids_probs_indices.tolist()[0])

                main_model_relative_representation_probs = torch.mm(main_model_generate_ids_probs,
                                                                    self.main_model_probability_transfer_matrix_list[
                                                                        0]).to(self.device_compute)
                main_model_relative_values, main_model_relative_indices = torch.topk(
                    main_model_relative_representation_probs, k=10)
                json_object[f'main_rel_values'] = main_model_relative_values.tolist()[0]
                json_object[f'main_rel_indices'] = main_model_relative_indices.tolist()[0]

                model_relative_representation_probs_list = [main_model_relative_representation_probs]

                for index, (assist_model_generate_ids_logits, assist_model_probability_transfer_matrix) in enumerate(
                        zip(
                                assist_model_generate_ids_logits_list,
                                self.assist_model_probability_transfer_matrix_list)):
                    assist_model_generate_ids_probs = nn.functional.softmax(assist_model_generate_ids_logits,
                                                                            dim=-1).float().to(
                        assist_model_probability_transfer_matrix.device)

                    values, indices = torch.topk(assist_model_generate_ids_probs, k=10)
                    json_object[f'origin_aux_{index}_top_tokens'] = self.assist_model_tokenizer_list[
                        index].convert_ids_to_tokens(indices.tolist()[0])

                    assist_model_relative_representation_probs = torch.mm(assist_model_generate_ids_probs,
                                                                          assist_model_probability_transfer_matrix).to(
                        self.device_compute)

                    assist_model_relative_values, assist_model_relative_indices = torch.topk(
                        assist_model_relative_representation_probs, k=10)
                    json_object[f'aux_rel_values_{index}'] = assist_model_relative_values.tolist()[0]
                    json_object[f'aux_rel_indices_{index}'] = assist_model_relative_indices.tolist()[0]

                    model_relative_representation_probs_list.append(assist_model_relative_representation_probs)

            json_object[f'ensemble_weight'] = self.ensemble_weight

            average_probs = torch.zeros_like(main_model_relative_representation_probs)
            # print(self.ensemble_weight)
            for weight, probs in zip(self.ensemble_weight, model_relative_representation_probs_list):
                average_probs += weight * probs

            average_relative_probs_values, average_relative_probs_indices = torch.topk(
                average_probs, k=10)

            json_object[f'average_rel_probs_values'] = average_relative_probs_values.tolist()[0]
            json_object[f'average_rel_probs_indices'] = average_relative_probs_indices.tolist()[0]

            torch.set_grad_enabled(True)

            main_model_generate_ids_logits = main_model_generate_ids_logits.to(self.device_compute).detach().clone().to(
                torch.float32)
            main_model_generate_ids_logits.requires_grad_(True)

            local_learning_rate = self.learning_rate
            criterion = nn.KLDivLoss()

            optimizer = torch.optim.AdamW(params=[main_model_generate_ids_logits],
                                          lr=local_learning_rate,
                                          betas=(0.9, 0.999))

            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=local_learning_rate / 4)

            for i in range(1, self.learning_epochs_nums + 1):
                main_model_generate_ids_probs = nn.functional.softmax(main_model_generate_ids_logits, dim=-1).float()
                main_model_relative_representation_probs = torch.mm(main_model_generate_ids_probs,
                                                                    self.main_model_probability_transfer_matrix_list[0])

                log_main_probs = torch.log(main_model_relative_representation_probs)
                loss = criterion(log_main_probs, average_probs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                main_model_generate_ids_logits_probs_values, main_model_generate_ids_logits_indices = torch.topk(
                    torch.nn.functional.softmax(main_model_generate_ids_logits, dim=-1), k=10)
                json_object[f'main_model_generate_ids_logits_probs_values_{i}'] = \
                    main_model_generate_ids_logits_probs_values.tolist()[0]
                json_object[f'main_model_generate_ids_logits_indices_{i}'] = \
                    self.main_model_tokenizer.convert_ids_to_tokens(
                        main_model_generate_ids_logits_indices.tolist()[0])

            torch.set_grad_enabled(False)
            optionA_ppl = self.calculate_ppl(main_model_generate_ids_logits.to(self.device), optionA)
            optionB_ppl = self.calculate_ppl(main_model_generate_ids_logits.to(self.device), optionB)

            ppl_ABCD_list.append(optionA_ppl)
            ppl_ABCD_list.append(optionB_ppl)
            next_tokens_id = token_ABCD_index_list[ppl_ABCD_list.index(min(ppl_ABCD_list))]

            main_model_generate_ids_logits_probs_values, main_model_generate_ids_logits_indices = torch.topk(
                torch.nn.functional.softmax(main_model_generate_ids_logits, dim=-1), k=10)
            json_object[f'main_model_generate_ids_logits_probs_values_final'] = \
                main_model_generate_ids_logits_probs_values.tolist()[0]
            json_object[f'main_model_generate_ids_logits_indices_final'] = \
                self.main_model_tokenizer.convert_ids_to_tokens(
                    main_model_generate_ids_logits_indices.tolist()[0])

            with open(ensemble_process_file_path, "a+", encoding="utf-8") as process_file:
                process_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')

            output = torch.zeros_like(scores).to(self.device)
            output[:, next_tokens_id] = float('inf')
            return output
        else:
            optionA_ppl = self.calculate_ppl(scores, optionA)
            optionB_ppl = self.calculate_ppl(scores, optionB)
            ppl_ABCD_list.append(optionA_ppl)
            ppl_ABCD_list.append(optionB_ppl)
            next_tokens_id = token_ABCD_index_list[ppl_ABCD_list.index(min(ppl_ABCD_list))]
            output = torch.zeros_like(scores).to(self.device)
            output[:, next_tokens_id] = float('inf')
            return output
