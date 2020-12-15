import torch

class REDataset:
    def __init__(self, tokens, tags, tokenizer):
        self.tokens = tokens
        self.relation_tags = tags
        self.tokenizer = tokenizer

        self.rel_tags = list(set(word_rel for sent in self.relation_tags for word_rel in sent))
        self.rel_tags = ["<PAD>"] + self.rel_tags
        self.rel_tag2idx = {tag: idx for idx, tag in enumerate(self.rel_tags)}
        self.rel_idx2tag = {idx: tag for idx, tag in enumerate(self.rel_tags)}

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        sentences, relations = [], []
        words = [word for word in self.tokens[idx] if word != '\uf0b7']
        tags = [word_relation for word_relation in self.relation_tags[idx]]
        sentences.append(["[CLS]"] + words + ["[SEP]"])
        relations.append(["<PAD>"] + tags + ["<PAD>"])
        tokens_ids, tags_ids = [], []

        for word, tag in zip(sentences[0], relations[0]):
            tokens = self.tokenizer.tokenize(word) if word not in ("[CLS]", "[SEP]") else [word]
            tokens_idx = self.tokenizer.convert_tokens_to_ids(tokens)

            tag = [tag] + ["<PAD>"] * (len(tokens) - 1)
            tags_idx = [self.rel_tag2idx[each] for each in tag]

            tokens_ids.extend(tokens_idx)
            tags_ids.extend(tags_idx)

        length = len(tags_ids)

        return tokens_ids, tags_ids, length

    def paddings(self, batch):
        max_length = max([length[2] for length in batch])
        tokens = [token[0] + [0] * (max_length - len(token[0])) for token in batch]
        tags = [tag[1] + [0] * (max_length - len(tag[1])) for tag in batch]

        return torch.LongTensor(tokens), torch.LongTensor(tags)

