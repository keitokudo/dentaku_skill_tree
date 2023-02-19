import argparse
from pathlib import Path
from itertools import islice
import pickle
import random
import shutil
import json

from more_itertools import ilen, chunked, ichunked
from logzero import logger
from tqdm import tqdm, trange

from numerical_data_generator import NumericDataGenarator
from dataset_generator_base import DatasetGeneratorBase


class DentakuDatasetGeneratorBase(DatasetGeneratorBase):
    def __init__(
            self,
            data_config_file_path,
            number_of_data:int,
            save_file_path:Path,
            exclude_dataset_paths=None,
    ):
        
        raise NotImplementedError()

    def formula_configs_loader(self):
        assert self.fconf_file_path.exists(), f"\"{self.fconf_file_path}\" is not exist..."
        with self.fconf_file_path.open(mode="rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
    


    


class DentakuDatasetGenerator(DentakuDatasetGeneratorBase):
    def __init__(
            self,
            data_config_file_path,
            number_of_data:int,
            save_file_path:Path,
            exclude_dataset_paths=None,
    ):
        
        self.save_file_path = save_file_path
        self.exclude_dataset_paths = [] if exclude_dataset_paths is None else exclude_dataset_paths
        self.exclude_dataset_sets = []
        
        for path in self.exclude_dataset_paths:
            with (path / "set.pkl").open(mode="rb") as f:
                self.exclude_dataset_sets.append(pickle.load(f))

        self.data_config_file_path = data_config_file_path
        self.question_generator = NumericDataGenarator(
            self.data_config_file_path
        )
        
        assert self.question_generator.max_number_of_question == "inf", "Question_generator config's \"max_number_of_question\" must be \"inf\""
        
        self.number_of_data = number_of_data
        self.passage_question_set = None
        self.formula_configs = None
        

    def prepare_data(self):
        self.save_file_path.mkdir(exist_ok=True)

        shutil.copy(
            self.data_config_file_path,
            self.save_file_path / self.data_config_file_path.name
        )
        
        # Save basic info
        self.basic_info_file_path = self.save_file_path / "basic_info.json"
        with self.basic_info_file_path.open(mode="w") as f:
            json.dump(
                {
                    "number_of_data": self.number_of_data,
                    "collator": "DataCollator",
                },   
                f
            )

        self.set_file_path = self.save_file_path / "set.pkl"
        self.fconf_file_path = self.save_file_path / "fconf.pkl"
        self.raw_data_file_path = self.save_file_path / "raw_data.tsv"
        self.passage_question_set = set()

        
        with self.fconf_file_path.open(mode="wb") as f_fconf, \
             self.raw_data_file_path.open(mode="w") as f_raw:
            
            for i in trange(self.number_of_data):
                while True:
                    operator_config, assignment_configs = next(
                        self.question_generator(generate_config=True)
                    )

                    pqa_triple_list = self.question_generator.get_pqa_triple_from_configs(
                        operator_config,
                        assignment_configs,
                        separate=True
                    )
                    assert len(pqa_triple_list) == 1, "Only support \"ask_last_question\" mode"
                    pqa_triple = pqa_triple_list[0]

                    if not (self.in_exclude_datasets(pqa_triple[:-1]) or self.in_myself(pqa_triple[:-1])):
                        break


                # Only passage and question
                self.passage_question_set.add(pqa_triple[:-1])
                pickle.dump((operator_config, assignment_configs), f_fconf)
                f_raw.write("\t".join(pqa_triple) + "\n")
        
        assert len(self.passage_question_set) == self.number_of_data, "Set size dosen't match..."
        with self.set_file_path.open(mode="wb") as f:
            pickle.dump(self.passage_question_set, f)
        
        assert ilen(self.formula_configs_loader()) == self.number_of_data, "formula_configs size dosen't match..."
