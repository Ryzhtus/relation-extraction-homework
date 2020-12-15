class REDataset:
    def __init__(self, parser, tokenizer):
        self.tokens, self.ner_tags, self.relation_tags = parser.extract_data()
        self.tokenizer = tokenizer

        self.rel_tags = list(set(word_rel for sent in self.relation_tags for word_rel in sent))
        self.rel_tags = ["<pad>"] + self.rel_tags
        self.rel_tag2idx = {tag: idx for idx, tag in enumerate(self.rel_tags)}
        self.rel_idx2tag = {idx: tag for idx, tag in enumerate(self.rel_tags)}

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        sentences, relations = [], []
        words = [word for word in self.tokens[idx] if word != '\uf0b7']
        tags = [word_rel for word_rel in self.relation_tags[idx]]
        sentences.append(["[CLS]"] + words + ["[SEP]"])
        relations.append(["<pad>"] + tags + ["<pad>"])
        x, y = [], []
        for w, t in zip(sentences[0], relations[0]):
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = self.tokenizer.convert_tokens_to_ids(tokens)

            t = [t] + ["<pad>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [self.rel_tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            y.extend(yy)

        assert len(x) == len(y), "words: {}, len(y)={}, x={}, y={}".format(words, len(y), x, y)

        # seqlen
        seqlen = len(y)

        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, tags, y, seqlen

