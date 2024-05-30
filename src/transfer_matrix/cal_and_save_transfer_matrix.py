import json
import os
import sys

import torch

from src.model_load import load_tokenizer, load_model_only
from src.transfer_matrix.common_vocabulary import CommonVocabulary
from src.transfer_matrix.transfer_matrix import ProbabilityTransferMatrix


def probability_transfer_matrix_save(model, anchor_point_index, temperature, device_compute, device, save_path):
    probability_transfer_matrix = probability_transfer_matrix_obj.get_final_probability_transfer_matrix(
        model=model, anchor_point_list=[anchor_point_index], temperature=temperature,
        device_compute=device_compute, device=device)
    torch.save(probability_transfer_matrix.float(), save_path)


# model_paths = [
#     "01-ai/Yi-6B",
#     "Skywork/Skywork-13B-base",
#     "mistralai/Mixtral-8x7B-v0.1",
#     "meta-llama/Llama-2-70b-hf",
#     "TigerResearch/tigerbot-13b-base-v2",
#     "mistralai/Mistral-7B-v0.1",
#     "internlm/internlm-20b",
#     "meta-llama/Llama-2-13b-hf"
# ]

probability_transfer_matrix_save_path = sys.argv[1] + "/"
model_paths = sys.argv[2:]
probability_transfer_matrix_name_list = [os.path.basename(model_path) for model_path in model_paths]
probability_transfer_matrix_save_path += "_".join(probability_transfer_matrix_name_list)
print("probability_transfer_matrix_save_path:", probability_transfer_matrix_save_path)
temperature = 100

tokenizers = [load_tokenizer(model_path) for model_path in model_paths]
vocab_lengths = [len(tokenizer.get_vocab()) for tokenizer in tokenizers]
print(vocab_lengths)

common_vocabulary = CommonVocabulary(*tokenizers)
common_vocab_list = common_vocabulary.get_common_vocab_list(*common_vocabulary.vocabs)
print(f"common_vocab_list: {len(common_vocab_list)}")

try:
    os.makedirs(probability_transfer_matrix_save_path)
except FileExistsError:
    pass

with open(f"{probability_transfer_matrix_save_path}/common_vocab_list.json", "w", encoding="utf8") as f:
    json.dump(common_vocab_list, f, ensure_ascii=False)

probability_transfer_matrix_obj = ProbabilityTransferMatrix()
anchor_point_index_list = probability_transfer_matrix_obj.get_anchor_point_list(common_vocab_list=common_vocab_list)

for index, model_path in enumerate(model_paths):
    model = load_model_only(model_path, "balanced_low_0")

    probability_transfer_matrix_save(
        model,
        anchor_point_index_list[index],
        temperature,
        device_compute="cuda:0",
        device="cuda:0",
        save_path=os.path.join(probability_transfer_matrix_save_path, f"{os.path.basename(model_path)}.pth")
    )
