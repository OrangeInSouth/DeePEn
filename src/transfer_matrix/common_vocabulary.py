import json
import pdb


class CommonVocabulary:
    def __init__(self, *tokenizers_and_paths):
        self.vocabs = self.get_vocabulary(*tokenizers_and_paths)

    def get_vocabulary(self, *tokenizers_and_paths):
        vocabs = []
        for item in tokenizers_and_paths:
            if isinstance(item, str):
                # If it's a vocab file path, get vocabulary by vocab file
                vocab = self.get_vocabulary_by_vocab_file(item)
                vocabs.append(vocab)
            else:
                # If it's a tokenizer, get vocabulary by tokenizer
                vocab = self.get_vocabulary_by_tokenizer(item)
                vocabs.append(vocab)
            # else:
            #     raise ValueError("Each item in the list should be either a tokenizer or a vocab file path")

        return vocabs

    def get_vocabulary1(self, tokenizer=None, vocab_file_path=None):
        if tokenizer:
            return self.get_vocabulary_by_tokenizer(tokenizer)
        elif vocab_file_path:
            return self.get_vocabulary_by_vocab_file(vocab_file_path)

    def find_common_elements(self, *args):
        # 使用set.intersection()找出所有列表的交集
        common_elements = set.intersection(*map(set, args))
        common_elements_list = list(common_elements)
        return common_elements_list

    def get_common_vocab_list(self, *vocabs):
        # 提取所有词典的 key
        keys_list = [set(vocab.keys()) for vocab in vocabs]

        # 使用set.intersection()找出所有列表的交集
        common_vocab_token_list = self.find_common_elements(*keys_list)

        # 构建结果列表
        common_vocab_list = [
            {
                "common_vocab_token": common_vocab_token,
                **{f"vocab_{i + 1}_index": vocab[common_vocab_token] for i, vocab in enumerate(vocabs)}
            }
            for common_vocab_token in common_vocab_token_list
        ]
        return common_vocab_list

    def get_vocabulary_by_tokenizer(self, tokenizer):
        vocab_dict = tokenizer.get_vocab()
        modified_dict = {}
        for key in vocab_dict:
            new_key = key.replace("Ġ", "▁")
            modified_dict[new_key] = vocab_dict[key]
        return modified_dict

    def get_vocabulary_by_vocab_file(self, vocab_file_path):
        with open(vocab_file_path, 'r', encoding='utf-8') as main_vocab_file:
            vocab_obj = json.loads(main_vocab_file.read())
            vocab_dict = vocab_obj["model"]["vocab"]
            return vocab_dict
