from pathlib import Path
from logzero import logger

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch_source):
        batch = {k:[] for k in batch_source[0].keys()}
        for instance in batch_source:
            for k, v in instance.items():
                batch[k].append(v)

        with self.tokenizer.as_target_tokenizer():
            label_batch = self.tokenizer.pad(
                {"input_ids": batch["labels"]},
                return_attention_mask=False
            )
            batch["labels"] = label_batch["input_ids"].masked_fill(
                label_batch["input_ids"] == self.tokenizer.pad_token_id,
                -100
            )

        return self.tokenizer.pad(batch, return_attention_mask=True, return_tensors="pt")
        
