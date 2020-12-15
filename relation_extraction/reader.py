class Reader:
    def __init__(self, data):
        self.data = data
        self.relations = []
        self.ner_tags = []

        self.rel2idx = dict()
        self.ner2idx = dict()

    def add_relation(self, relation_id, relation_type, start_idx, end_idx):
        relation_idx = len(self.relations)
        self.rel2idx[relation_id] = relation_idx
        self.relations.append((relation_type, start_idx, end_idx))

    def add_ner(self, ner_id, ner_type, head_l, head_r):
        ner_idx = len(self.ner_tags)
        self.ner2idx[ner_id] = ner_idx
        self.ner_tags.append((ner_type, head_l, head_r))
