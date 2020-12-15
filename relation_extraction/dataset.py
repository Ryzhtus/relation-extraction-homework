import torch
from torch.nn.utils.rnn import pad_sequence

class REDataset:
    def __init__(self, parser, tokenizer):
        self.tokens, self.ner_tags, self.relation_tags = parser.extract_data()
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

        return torch.LongTensor(tokens_ids), torch.LongTensor(tags_ids)

    def paddings(self, batch):
        tokens, tags = list(zip(*batch))

        tokens = pad_sequence(tokens, batch_first=True)
        tags = pad_sequence(tags, batch_first=True)

        return tokens, tags