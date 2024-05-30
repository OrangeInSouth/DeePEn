def answer_extract(text, demon_count, split_key_before_list, split_key_behind_list):
    model_answer = text.split(split_key_before_list[0])[demon_count + 1:]
    try:
        prediction = model_answer[0]
        if len(split_key_before_list) > 1:
            try:
                prediction = prediction.split(split_key_before_list[1])[-1]
            except:
                pass
        for split_key_behind in split_key_behind_list:
            if split_key_behind in prediction:
                prediction = prediction.split(split_key_behind)[0]
    except:
        prediction = ""
    return model_answer, prediction
