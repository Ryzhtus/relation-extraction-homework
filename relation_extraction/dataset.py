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

            tag = [tag] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
            tags_idx = [self.rel_tag2idx[each] for each in tag]  # (T,)

            tokens_ids.extend(tokens_idx)
            tags_ids.extend(tags_idx)

        assert len(tokens_ids) == len(tags_ids), "words: {}, len(tags_ids)={}, tokens_ids={}, tags_ids={}".format(words, len(tags_ids), tokens_ids, tags_ids)

        sequential_length = len(tags_ids)

        words = " ".join(words)
        tags = " ".join(tags)
        return words, tokens_ids, tags, tags_ids, sequential_length

