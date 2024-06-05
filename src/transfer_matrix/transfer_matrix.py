import torch
from tqdm import tqdm


class ProbabilityTransferMatrix():
    def get_anchor_point_list(self, common_vocab_list):
        # [{'common_vocab_token': '▁What', 'vocab_1_index': 1724, 'vocab_2_index': 1824, 'vocab_3_index': 2371}]

        vocab_keys = [key for key in common_vocab_list[0].keys() if key.startswith('vocab_')]
        index_lists = [[common_vocab.get(key, None) for common_vocab in common_vocab_list] for key in vocab_keys]

        return index_lists

    def get_final_probability_transfer_matrix(self, model,
                                              anchor_point_list, device_compute,
                                              temperature=100,
                                              device="cuda:0"):
        model_relative_representation_matrix = self.get_relative_representation_matrix_by_cosine_similarity_normalization(
            model, anchor_point_list, device_compute, device)

        model_probability_transfer_matrix = self.get_probability_transfer_matrix(model_relative_representation_matrix,
                                                                                 temperature)

        return model_probability_transfer_matrix

    def block_cosine_similarity(self, tensor1, tensor2, block_size=1000):
        with torch.no_grad():
            size1 = tensor1.size()
            size2 = tensor2.size()
            result = torch.zeros(size1[0], size2[0])
            for i in tqdm(range(0, size1[0], block_size)):
                for j in range(0, size2[0], block_size):
                    result[i:i + block_size, j:j + block_size] = torch.cosine_similarity(
                        tensor1[i:i + block_size].unsqueeze(1), tensor2[j:j + block_size].unsqueeze(0), dim=-1)

            return result

    def cosine_similarity(self, tensor1, tensor2):
        with torch.no_grad():
            return torch.cosine_similarity(tensor1.unsqueeze(1), tensor2.unsqueeze(0), dim=-1)

    def get_relative_representation_matrix_by_cosine_similarity_normalization(self, model, anchor_point_list_index,
                                                                              device_compute, device):

        model_anchor_point_index_list = anchor_point_list_index

        model_embedding_tensor = model.get_input_embeddings().weight.to(device_compute)
        model_anchor_point_embedding_tensor = model_embedding_tensor[model_anchor_point_index_list].to(
            device_compute)

        block_size = 1000

        model_relative_representation_matrix = self.block_cosine_similarity(
            model_embedding_tensor,
            model_anchor_point_embedding_tensor,
            block_size
        )

        model_relative_representation_matrix = torch.nn.functional.normalize(
            model_relative_representation_matrix, p=2, dim=-1).to(device)

        return model_relative_representation_matrix

    def get_probability_transfer_matrix(self, model_relative_representation_matrix, temperature):

        torch.cuda.empty_cache()
        # probability_transfer_matrix = torch.nn.functional.softmax(
        #     model_relative_representation_matrix * temperature, dim=-1)
        probability_transfer_matrix = self.softmax_by_chunk(model_relative_representation_matrix, temperature, 10000)
        torch.cuda.empty_cache()

        return probability_transfer_matrix

    def softmax_by_chunk(self, model_relative_representation_matrix, temperature, chunk_size=2048, device="cpu"):
        num_rows, num_cols = model_relative_representation_matrix.shape
        probability_transfer_matrix = torch.empty_like(model_relative_representation_matrix, device=device)

        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            chunk = model_relative_representation_matrix[start:end]
            chunk_softmax = torch.nn.functional.softmax(chunk * temperature, dim=-1)
            probability_transfer_matrix[start:end] = chunk_softmax

        return probability_transfer_matrix

    def get_blocked_probability_transfer_matrix(self, model_relative_representation_matrix, temperature, block_size=16):

        input_shape = model_relative_representation_matrix.shape
        reshaped_matrix = model_relative_representation_matrix.view(-1, block_size, input_shape[-1])
        reshaped_matrix = reshaped_matrix * temperature
        softmaxed_blocks = torch.nn.functional.softmax(reshaped_matrix, dim=-1)
        probability_transfer_matrix = softmaxed_blocks.view(input_shape)
        return probability_transfer_matrix

    def get_anchor_point_list_our(self, main_model_embedding_tensor, assist_model_embedding_tensor,
                                  common_vocab_list, count):
        main_model_common_vocab_index_list = [common_vocab['vocab_1_index'] for common_vocab in
                                              common_vocab_list]
        assist_model_common_vocab_index_list = [common_vocab['vocab_2_index'] for common_vocab in
                                                common_vocab_list]

        model_common_vocab_token_list = [common_vocab['common_vocab_token'] for common_vocab in common_vocab_list]

        main_model_anchor_point_embedding = main_model_embedding_tensor[main_model_common_vocab_index_list]

        assist_model_anchor_point_embedding = assist_model_embedding_tensor[assist_model_common_vocab_index_list]

        block_size = 1000
        main_model_common_vocab_self_cosine_similarity_matrix = self.block_cosine_similarity(
            main_model_anchor_point_embedding,
            main_model_anchor_point_embedding,
            block_size)

        assist_model_common_vocab_self_cosine_similarity_matrix = self.block_cosine_similarity(
            assist_model_anchor_point_embedding,
            assist_model_anchor_point_embedding,
            block_size)

        # 计算每个共享词的跨模型相对表示一致性
        cross_model_relative_representation_matrix_consistency = torch.cosine_similarity(
            main_model_common_vocab_self_cosine_similarity_matrix,
            assist_model_common_vocab_self_cosine_similarity_matrix,
            dim=-1)

        cross_model_relative_representation_matrix_consistency_list = cross_model_relative_representation_matrix_consistency.tolist()

        sorted_sim_tuples = sorted(enumerate(cross_model_relative_representation_matrix_consistency_list),
                                   key=lambda x: x[1], reverse=True)

        # assert len(
        #     common_vocab_list) > 1000, "Please make sure the models to ensemble have more than 1,000 common words"

        top_k_indices = [x[0] for x in sorted_sim_tuples]

        # optimal_anchor_num = 1000
        # optimal_consistency = -1
        # for trial_anchor_num in range(1000, len(common_vocab_list), 1000):
        #     trial_anchor_list = top_k_indices[:trial_anchor_num]
        #     main_relative_representation_common, aux_relative_representation_common = self.get_relative_representation_of_common_vocab(
        #         main_model_common_vocab_self_cosine_similarity_matrix,
        #         assist_model_common_vocab_self_cosine_similarity_matrix,
        #         trial_anchor_list)
        #     trial_res = self.cal_consistency(main_relative_representation_common, aux_relative_representation_common)
        #     if trial_res > optimal_consistency:
        #         optimal_consistency = trial_res
        #         optimal_anchor_num = trial_anchor_num
        trial_anchor_num = count
        trial_anchor_list = top_k_indices[:trial_anchor_num]
        main_relative_representation_common, aux_relative_representation_common = self.get_relative_representation_of_common_vocab(
            main_model_common_vocab_self_cosine_similarity_matrix,
            assist_model_common_vocab_self_cosine_similarity_matrix,
            trial_anchor_list)
        optimal_anchor_num = trial_anchor_num
        optimal_consistency = self.cal_consistency(main_relative_representation_common,
                                                   aux_relative_representation_common)

        print(f"Optimal Anchor Num: {optimal_anchor_num}")
        print(f"Optimal Consistency between Relative Embeddings: {optimal_consistency}")
        # pdb.set_trace()
        anchor_point_list = []
        for index in top_k_indices[:optimal_anchor_num]:
            dict = {}
            dict['vocab_1_index'] = main_model_common_vocab_index_list[index]
            dict['vocab_2_index'] = assist_model_common_vocab_index_list[index]
            dict['common_vocab_token'] = model_common_vocab_token_list[index]
            anchor_point_list.append(dict)

        return anchor_point_list

    def cal_optimal_consistency(self, main_model_embedding_tensor, assist_model_embedding_tensor,
                                model_common_vocab_index_list):
        main_model_common_vocab_index_list = model_common_vocab_index_list[0]
        assist_model_common_vocab_index_list = model_common_vocab_index_list[1]
        common_vocab_list = main_model_common_vocab_index_list
        main_model_anchor_point_embedding = main_model_embedding_tensor[main_model_common_vocab_index_list]

        assist_model_anchor_point_embedding = assist_model_embedding_tensor[assist_model_common_vocab_index_list]

        block_size = 1000
        main_model_common_vocab_self_cosine_similarity_matrix = self.block_cosine_similarity(
            main_model_anchor_point_embedding,
            main_model_anchor_point_embedding,
            block_size)

        assist_model_common_vocab_self_cosine_similarity_matrix = self.block_cosine_similarity(
            assist_model_anchor_point_embedding,
            assist_model_anchor_point_embedding,
            block_size)

        # 计算每个共享词的跨模型相对表示一致性
        cross_model_relative_representation_matrix_consistency = torch.cosine_similarity(
            main_model_common_vocab_self_cosine_similarity_matrix,
            assist_model_common_vocab_self_cosine_similarity_matrix,
            dim=-1)

        cross_model_relative_representation_matrix_consistency_list = cross_model_relative_representation_matrix_consistency.tolist()

        sorted_sim_tuples = sorted(enumerate(cross_model_relative_representation_matrix_consistency_list),
                                   key=lambda x: x[1], reverse=True)

        top_k_indices = [x[0] for x in sorted_sim_tuples]

        optimal_anchor_num = 1000
        optimal_consistency = -1

        trial_anchor_num = len(common_vocab_list)
        trial_anchor_list = top_k_indices[:trial_anchor_num]
        main_relative_representation_common, aux_relative_representation_common = self.get_relative_representation_of_common_vocab(
            main_model_common_vocab_self_cosine_similarity_matrix,
            assist_model_common_vocab_self_cosine_similarity_matrix,
            trial_anchor_list)
        trial_res = self.cal_consistency(main_relative_representation_common, aux_relative_representation_common)
        if trial_res > optimal_consistency:
            optimal_consistency = trial_res
            optimal_anchor_num = trial_anchor_num

        print(f"Optimal Anchor Num: {optimal_anchor_num}")
        print(f"Optimal Consistency between Relative Embeddings: {optimal_consistency}")
        # pdb.set_trace()

    def get_relative_representation_of_common_vocab(self, main_common_vocab_self_similarity_matrix,
                                                    aux_common_vocab_self_similarity_matrix, selected_indices):
        # pdb.set_trace()
        main_model_relative_representation_matrix = main_common_vocab_self_similarity_matrix[
            selected_indices].transpose(0,
                                        1)
        assist_model_relative_representation_matrix = aux_common_vocab_self_similarity_matrix[
            selected_indices].transpose(0,
                                        1)

        return main_model_relative_representation_matrix, assist_model_relative_representation_matrix

    def cal_consistency(self, main_relative_representation, aux_relative_representation):
        return torch.cosine_similarity(main_relative_representation, aux_relative_representation).mean().item()
