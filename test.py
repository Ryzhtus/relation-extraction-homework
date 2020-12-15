from relation_extraction.parser import Parser
from relation_extraction.dataset import REDataset
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

parser = Parser(tokenizer, 'data/train/*.ann')

dataset = REDataset(parser, tokenizer)
print(dataset[0])
