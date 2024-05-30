import torch
from torch import nn
from tqdm import tqdm


def softmax_with_temperature(logits, temperature):
    logits = logits / temperature
    return nn.functional.softmax(logits, dim=-1)


def block_cosine_similarity(tensor1, tensor2, block_size=100):
    with torch.no_grad():
        size1 = tensor1.size()
        size2 = tensor2.size()
        result = torch.zeros(size1[0], size2[0])
        for i in tqdm(range(0, size1[0], block_size)):
            for j in range(0, size2[0], block_size):
                result[i:i + block_size, j:j + block_size] = torch.cosine_similarity(
                    tensor1[i:i + block_size].unsqueeze(1), tensor2[j:j + block_size].unsqueeze(0), dim=-1)
        torch.cuda.empty_cache()
        return result
