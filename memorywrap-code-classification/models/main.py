import torch
from transformers import AutoTokenizer, AutoModel


class CodeEncoder:


    def __init__(self, model_name="bert-base-uncased", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        #not training encoder right now
        self.model.eval()

    def encode(self, text):
       

        # tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding

    def encode_batch(self, texts):
       
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        return cls_embeddings