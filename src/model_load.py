from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, AutoModelForSeq2SeqLM


def load_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map=device,
                                                 torch_dtype="auto", trust_remote_code=True)
    model = model.eval()
    tokenizer = load_tokenizer(model_path)
    # tokenizer.pad_token = tokenizer.eos_token
    streamer = TextStreamer(tokenizer)
    return model, tokenizer, streamer


def load_model_only(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map=device,
                                                 torch_dtype="auto", trust_remote_code=True)
    model = model.eval()
    return model


def load_model_nllb(model_path, device):
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, use_auth_token=True, device_map=device,
                                                  torch_dtype="auto", trust_remote_code=True)

    # model = AutoModelForCausalLM.from_pretrained(model_path,
    #                                              device_map=device,
    #                                              torch_dtype="auto", trust_remote_code=True)
    model = model.eval()
    return model


def load_tokenizer(main_model_path):
    tokenizer = AutoTokenizer.from_pretrained(main_model_path, padding_side='left', truncation_side='left',
                                              trust_remote_code=True)

    return tokenizer


def load_tokenizer_by_remote(main_model_path):
    tokenizer = AutoTokenizer.from_pretrained(main_model_path, padding_side='left', truncation_side='left',
                                              trust_remote_code=True)

    return tokenizer

    # if tokenizer.pad_token_id is None:
    #     if tokenizer.eos_token_id is not None:
    #         tokenizer.pad_token_id = tokenizer.eos_token_id
    #     else:
    #         tokenizer.pad_token_id = 0
