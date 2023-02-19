from pathlib import Path
import pickle

from logzero import logger
from tqdm import tqdm


class DatasetPreProcessor:
    def __init__(self, tokenizer, dataset_path:Path):
        self.tokenizer = tokenizer
        assert dataset_path.is_dir(), "f{dataset_path} is not directory..."
        self.dataset_path = dataset_path
        self.raw_data_file_path = self.dataset_path / "raw_data.tsv"
        self.tokenized_data_file_path = self.dataset_path / "tokenized_data.pkl"
        
    def make_tokenized_data_from_file(self):        
        assert self.raw_data_file_path.exists(), f"{self.raw_data_file_path} is not exists..."

        with self.raw_data_file_path.open("r") as f_raw:
            number_of_sequence = len(next(f_raw).strip().split("\t"))
            
        with self.raw_data_file_path.open("r") as f_raw, \
             self.tokenized_data_file_path.open("wb") as f_tok:
            
            for instance in tqdm(map(lambda s: s.strip().split("\t"), f_raw)):
                assert len(instance) == number_of_sequence, \
                    f"number_of_sequence({number_of_sequence}) in first line and {instance} differ..."
                
                pickle.dump(
                    self.make_model_input(instance),
                    f_tok
                )
        

    def make_model_input(self, instance):
        tokenized_input = self.tokenizer(*instance[:-1], padding=False, truncation=True, return_tensors='pt')
        assert all(len(v) == 1 for v in tokenized_input.values()), \
            "Too many elements exist in tokenized_input"
        tokenized_input = {k:v[0] for k, v in tokenized_input.items()}
        
        with self.tokenizer.as_target_tokenizer():
            tokenized_answers = self.tokenizer(
                instance[-1],
                padding=False,
                truncation=True,
                return_tensors='pt'
            )

        assert len(tokenized_answers["input_ids"]) == 1, \
            "Too many elements exist in tokenized_answers"    
        tokenized_input["labels"] = tokenized_answers["input_ids"][0]
        return tokenized_input

