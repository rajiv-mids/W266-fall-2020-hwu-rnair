import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from transformers import BertModel, BertConfig
import torch

class Bert(nn.Module):
    def __init__(self, temp_dir="/tmp/bert", load_pretrained_bert=True, bert_config=None):
        super(Bert, self).__init__()
        print(temp_dir)
        if(load_pretrained_bert):
            self.model = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = BertModel(bert_config)

    def forward(self, x, segs, mask):
        # the below returns a tuple. First element in the tuple is last hidden state. Second element in tuple is pooler output
        result = self.model(x, attention_mask =mask, position_ids=segs)
#        top_vec = encoded_layers[-1]

        top_vec = result[0]
        return top_vec


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x)
        h = h.squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class Summarizer(nn.Module):
    def __init__(self, args=None, num_hidden = 768, load_pretrained_bert = True, bert_config = None):
        super(Summarizer, self).__init__()
        self.args = args
        self.bert = Bert( load_pretrained_bert, bert_config=None)
#        if (args.encoder == 'classifier'):
        self.encoder = Classifier(num_hidden)

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):
        top_vec = self.bert(x, segs, mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls



