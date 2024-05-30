import torch
from transformers import LogitsProcessor


class AssistModelLogitsProcessor(LogitsProcessor):
    def __init__(self, assist_logits_queue, assist_result_file_path):
        self.assist_logits_queue = assist_logits_queue
        self.assist_result_file_path = assist_result_file_path
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.assist_logits_queue.put(scores)
        return scores
