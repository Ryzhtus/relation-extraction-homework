from transformers import BertModel
import torch.nn as nn

class CustomBert(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout_level, num_layers=1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = 768
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout_level = dropout_level

        self.bert = BertModel.from_pretrained("distilbert-base-multilingual-cased")

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers,
                            bidirectional=True, batch_first=True)

        self.rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim)

        self.linear = nn.Linear(self.hidden_dim, self.num_classes)

        self.dropout = nn.Dropout(self.dropout_level)

    def forward(self, tokens):
        tokens = tokens.permute(1, 0)

        embeddings = self.bert(tokens)[0]
        embeddings_drop = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings_drop)
        lstm_out = lstm_out.permute(1, 0, 2)
        lstm_drop = self.dropout(lstm_out)

        rnn_out, _ = self.rnn(lstm_drop)

        predictions = self.linear(rnn_out)

        return predictions