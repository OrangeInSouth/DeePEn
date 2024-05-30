import threading

from transformers import LogitsProcessorList

from src.logits_processor.assist_model_logits_processor import AssistModelLogitsProcessor


class AssistModelThread(threading.Thread):
    def __init__(self, model, model_tokenizer, assist_model_input, assist_model_score_queue, device, result_save_dir):
        self.model = model
        self.model_tokenizer = model_tokenizer
        self.model_input = assist_model_input
        self.model_scores_queue = assist_model_score_queue
        self.device = device
        self.result_save_dir = result_save_dir
        super().__init__()

    def run(self) -> None:
        model_input_ids = self.model_tokenizer(self.model_input, return_tensors="pt",
                                               add_special_tokens=False).input_ids.to(self.device)
        assist_model_generation_kwargs = {
            "input_ids": model_input_ids,
            "max_new_tokens": 1,
            "do_sample": False,
            "num_beams": 1,
            "eos_token_id": self.model_tokenizer.eos_token_id,
            "bos_token_id": self.model_tokenizer.bos_token_id,
            # "pad_token_id": self.model_tokenizer.pad_token_id
        }
        assist_model_logits_processor_list = LogitsProcessorList()
        assist_model_logits_processor_list.append(
            AssistModelLogitsProcessor(self.model_scores_queue, self.result_save_dir))

        assist_model_generate_ids = self.model.generate(**assist_model_generation_kwargs,
                                                        pad_token_id=self.model_tokenizer.eos_token_id,
                                                        logits_processor=assist_model_logits_processor_list)
        # print(assist_model_generate_ids)
        # print(self.model_tokenizer.decode(assist_model_generate_ids.tolist()[0]))
