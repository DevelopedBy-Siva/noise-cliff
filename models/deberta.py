import logging
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DebertaV2TokenizerFast,
    DebertaV2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm
from config import DEBERTA_CONFIG

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class DebertaModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DebertaV2TokenizerFast.from_pretrained(
            DEBERTA_CONFIG["model_name"]
        )
        self.model = None

    def _tokenize(self, texts):
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=DEBERTA_CONFIG["max_length"],
        )

    def fit(self, texts, labels):
        encodings = self._tokenize(list(texts))
        dataset = TextDataset(encodings, list(labels))
        loader = DataLoader(
            dataset,
            batch_size=DEBERTA_CONFIG["batch_size"],
            shuffle=True,
        )

        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            DEBERTA_CONFIG["model_name"],
            num_labels=2,
        ).to(self.device)

        optimizer = AdamW(
            self.model.parameters(),
            lr=DEBERTA_CONFIG["learning_rate"],
        )

        total_steps = len(loader) * DEBERTA_CONFIG["epochs"]
        warmup_steps = int(total_steps * DEBERTA_CONFIG["warmup_ratio"])

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.model.train()
        for epoch in range(DEBERTA_CONFIG["epochs"]):
            loop = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)
            for batch in loop:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                labels_batch = batch["labels"].to(self.device)

                inputs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch,
                )
                if token_type_ids is not None:
                    inputs["token_type_ids"] = token_type_ids.to(self.device)

                outputs = self.model(**inputs)
                outputs.loss.backward()
                optimizer.step()
                scheduler.step()
                loop.set_postfix(loss=outputs.loss.item())

        return self

    def predict_proba(self, texts):
        encodings = self._tokenize(list(texts))
        dataset = TextDataset(encodings, [0] * len(texts))
        loader = DataLoader(
            dataset,
            batch_size=DEBERTA_CONFIG["batch_size"],
            shuffle=False,
        )

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids")

                inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
                if token_type_ids is not None:
                    inputs["token_type_ids"] = token_type_ids.to(self.device)

                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.extend(probs)

        return np.array(all_probs)

    def predict(self, texts):
        proba = self.predict_proba(texts)
        return np.argmax(proba, axis=1).tolist()

    def get_loss_per_sample(self, texts, labels):
        """Per-sample log loss. Higher means the model is more confused about that sample."""
        proba = self.predict_proba(texts)
        eps = 1e-9
        losses = []
        for i, label in enumerate(labels):
            p = np.clip(proba[i][label], eps, 1 - eps)
            losses.append(-np.log(p))
        return losses

    def save(self, path):
        """Saves model and tokenizer to a directory."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path):
        """Loads a saved model from disk and returns a ready-to-use DebertaModel."""
        instance = cls.__new__(cls)
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.tokenizer = DebertaV2TokenizerFast.from_pretrained(path)
        instance.model = DebertaV2ForSequenceClassification.from_pretrained(path).to(
            instance.device
        )
        instance.model.eval()
        return instance
