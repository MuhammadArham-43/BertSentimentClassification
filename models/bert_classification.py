from typing import List
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertLMHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BertLMHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        logits = self.fc(self.dropout(x))
        return self.softmax(logits)

class BertSentimentClassifier(nn.Module):
    def __init__(self, num_classes: int, model_path: str, freeze_bert: bool = True):
        super(BertSentimentClassifier, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.bert = BertModel.from_pretrained(model_path)
        self.decoder = BertLMHead(self.bert.config.hidden_size, num_classes)
                
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def encode_texts(self, texts: List[str]):
        return self.tokenizer(texts, return_tensors="pt", add_special_tokens=True, padding="max_length", truncation=True, max_length=512)
    
    def forward(self, texts: torch.Tensor):
        bert_embedding = self.bert(**texts).pooler_output
        return self.decoder(bert_embedding)