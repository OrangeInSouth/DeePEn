from src.logits_processor.based_on_probaility_transfer_logits_processor import \
    BasedOnProbabilityTransferLogits_Loacal_FP32_Processor, \
    BasedOnProbabilityTransferLogits_Loacal_FP32_digit_vote_Processor
from src.logits_processor.ppl_based_on_probaility_transfer_logits_processor import \
    YiPPLBasedOnProbabilityTransferLogitsProcessor, \
    InternLMPPLBasedOnProbabilityTransferLogitsProcessor


# YiPPLBasedOnProbabilityTransferLogitsPIQAProcessor, \

class ModelProcessorFactory():
    @staticmethod
    def create_processor(processor_type, **kwargs):
        processor_classes = {

            "based_on_probility_transfer_logits_fp32_processor": BasedOnProbabilityTransferLogits_Loacal_FP32_Processor,
            "based_on_probility_transfer_logits_fp32_digit_vote_processor": BasedOnProbabilityTransferLogits_Loacal_FP32_digit_vote_Processor,

            "yi_ppl_based_on_probility_transfer_logits_processor": YiPPLBasedOnProbabilityTransferLogitsProcessor,
            "intermlm_ppl_based_on_probility_transfer_logits_processor": InternLMPPLBasedOnProbabilityTransferLogitsProcessor,

            # "yi_ppl_based_on_probility_transfer_logits_piqa_processor": YiPPLBasedOnProbabilityTransferLogitsPIQAProcessor

        }
        selected_processor_class = processor_classes.get(processor_type.lower())
        if selected_processor_class:
            return selected_processor_class(**kwargs)
        else:
            print(f"Unsupported processor type: {processor_type}")
            return None
