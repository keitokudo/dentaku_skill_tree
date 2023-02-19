import argparse
from pathlib import Path


import dentaku_t5_tokenizer
import dentaku_bart_tokenizer
import dentaku_dataset_generator
from dentaku_dataset_generator import (
    DentakuDatasetGenerator,
)
from preprocessors import DatasetPreProcessor

def main(args):
    args.save_file_path.mkdir(parents=True, exist_ok=True)
    
    generator_cls = getattr(dentaku_dataset_generator, args.dataset_generator_name)
    try:
        tokenizer_cls = getattr(dentaku_t5_tokenizer, args.tokenizer)
    except AttributeError:
        tokenizer_cls = getattr(dentaku_bart_tokenizer, args.tokenizer)
    tokenizer = tokenizer_cls.from_pretrained(args.model_name_or_path)

        
    if issubclass(generator_cls, DentakuDatasetGenerator):
        if not args.only_picklies:
            generator = generator_cls(
                args.data_config_file_path,
                args.number_of_data,
                args.save_file_path,
                args.exclude_dataset_paths,
            )
            generator.prepare_data()

        
        preprocessor = DatasetPreProcessor(
            tokenizer,
            args.save_file_path,
        )
        preprocessor.make_tokenized_data_from_file()

    else:
        raise NotImplementedError()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_generator_name", help="Select dataset generator name", type=str, required=True)
    parser.add_argument("--data_config_file_path", help="Select data_config_file_path", type=Path)
    parser.add_argument("--edit_config_file_path", help="Select edit_config_file_path", type=Path)
    parser.add_argument("--original_data_set_file_path", help="Select original_data_set_file_path", type=Path)
    parser.add_argument("--number_of_data", help="Select number of data per epoch", type=int)
    parser.add_argument("--exclude_dataset_paths", help="Select exclude_dataset_paths", nargs='*', type=Path, default=[])
    parser.add_argument("--save_file_path", help="Select save file path", type=Path, required=True)


    parser.add_argument("--tokenizer", help="Select tokenizer", type=str)
    parser.add_argument("--model_name_or_path", help="Select model name or path", type=str)

    parser.add_argument("--data_tag", help="Select test data tag", type=str)
    parser.add_argument("--only_picklies", help="Specify whether only only_picklies", action="store_true")
    
    args = parser.parse_args()
    
    main(args)
