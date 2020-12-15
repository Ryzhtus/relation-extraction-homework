import codecs
import re
import os
import glob

from relation_extraction.reader import Reader

import tqdm

import razdel
from nltk.tokenize.util import align_tokens

class Parser:
    def __init__(self, tokenizer, folder_path):
        self.folder_paths = glob.glob(folder_path)
        self.tokenizer = tokenizer

    def span_sentences(self, text, shift=0):
        sentences = [sentence.text for sentence in razdel.sentenize(text)]
        spans = align_tokens(sentences, text)
        spans = [(start + shift, end + shift) for start, end in spans]

        return sentences, spans

    def span_tokens(self, text, shift=0):
        tokens, spans = [], []

        for token in re.finditer(r"([^\W_]+|\S)", text):
            tokens.append(token.group(1))
            spans.append((shift + token.start(1), shift + token.end(1)))

        return tokens, spans

    def read_file(self, folder_path):
        with codecs.open(folder_path, encoding="utf-8") as input_file:
            data = input_file.readlines()

        text = ""
        if os.path.exists(folder_path[:-3] + "txt"):
            with codecs.open(folder_path[:-3] + "txt", encoding="utf-8") as input_file:
                text = input_file.read()

        ignored = dict()

        reader = Reader(text)
        ner_list = []
        for line_num, line in enumerate(data):
            line = line.strip()
            if line.startswith("T"):
                try:
                    arr = line.split('\t')
                    ner_id = arr[0]
                    ner = arr[1]
                except:
                    print("Invalid relation format")

                ner_type, start_idx, end_idx = ner.split()

                ner_id = int(ner_id[1:])
                start_idx = int(start_idx)
                end_idx = int(end_idx)

                ner_list.append((ner_id, ner_type, start_idx, end_idx))

        ner_list.sort(key=lambda x: x[2])
        for (ner_id, ner_type, start_idx, end_idx) in ner_list:
            reader.add_ner(ner_id, ner_type, start_idx, end_idx)

        for line_num, line in enumerate(data):
            line = line.strip()
            if line.startswith("R"):
                relation_id, rel = line.split('\t')
                relation_type, arg1, arg2 = rel.split()

                relation_id = int(relation_id[1:])
                head_l = int(arg1[6:])
                head_r = int(arg2[6:])

                if head_l in ignored:
                    if head_l in ignored[head_l]:
                        continue

                if head_r in ignored:
                    if head_r in ignored[head_r]:
                        continue

                reader.add_relation(relation_id, relation_type, head_l, head_r)

        return reader

    def extract_data(self):
        tokens, ner_tags, relation_tags = [], [], []

        for file_path in tqdm.tqdm(self.folder_paths):
            data = self.read_file(file_path)
            """data_ner_tags = [{"id": i, "ner_type": data.ner_tags[idx][0],
                              "start": data.ner_tags[idx][1],
                              "end": data.ner_tags[idx][2]} for i, idx in data.ner2idx.items()]"""

            data_relations = [{'id': i, 'relation_type': data.relations[idx][0],
                               'start_end_ne_1': data.ner_tags[data.ner2idx[data.relations[idx][1]]],
                               'start_end_ne_2': data.ner_tags[data.ner2idx[data.relations[idx][2]]]} for i, idx in data.rel2idx.items()]

            for line in re.finditer(r"[^\n]+(\n+|$)", data.data):
                sentences, sentences_spans = self.span_sentences(line.group(0), shift=line.start())

                for sent, (sent_start, _) in zip(sentences, sentences_spans):
                    token, spans = self.span_tokens(sent, shift=sent_start)
                    tokens.append(token)
                    #ner_tags.append(self.convert_to_conll_ner(data_ner_tags, spans))
                    relation_tags.append(self.convert_to_conll_relation(data_relations, spans))

        return tokens, relation_tags #ner_tags,

    """
    @staticmethod
    def convert_to_conll_ner(ner_tags, spans):
        conll_ners = []

        for token_start, token_end in spans:

            for ner in ner_tags:

                if (ner["start"] <= token_start) and (ner["end"] >= token_end):
                    prefix = "I" if (ner["start"] < token_start) else "B"
                    conll_ners.append(prefix + "-" + ner["ner_type"])
                    break

            else:
                conll_ners.append("O")

        return conll_ners"""

    @staticmethod
    def convert_to_conll_relation(relations, spans):
        conll_relations = []
        for token_start, token_end in spans:
            for relation in relations:
                start_ne_1 = relation["start_end_ne_1"][1]
                end_ne_1 = relation["start_end_ne_1"][2]

                start_ne_2 = relation["start_end_ne_2"][1]
                end_ne_2 = relation["start_end_ne_2"][2]

                if start_ne_1 > start_ne_2:
                    start_ne_2 = relation["start_end_ne_1"][1]
                    end_ne_2 = relation["start_end_ne_1"][2]

                    start_ne_1 = relation["start_end_ne_2"][1]
                    end_ne_1 = relation["start_end_ne_2"][2]

                if ((start_ne_1 <= token_start) and (end_ne_1 >= token_end)) or (
                        (start_ne_2 <= token_start) and (end_ne_2 >= token_end)):
                    conll_relations.append(relation["relation_type"])
                    break
            else:
                conll_relations.append("O")
        return conll_relations