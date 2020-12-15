from relation_extraction.parser import Parser
from relation_extraction.dataset import REDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader

tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

parser = Parser(tokenizer, 'data/train/*.ann')

dataset = REDataset(parser, tokenizer)

train_iterator = DataLoader(dataset=dataset,
                             batch_size=16,
                             shuffle=True,
                             collate_fn=dataset.paddings)

print(next(iter(train_iterator)))