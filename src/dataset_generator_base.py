from pathlib import Path

class DatasetGeneratorBase():
    def __init__(
            self,
            number_of_data:int,
            save_file_path:Path,
            exclude_dataset_paths=None,
    ):
        
        raise NotImplementedError()
        
    def prepare_data(self):        
        raise NotImplementedError()

    
    def in_exclude_datasets(self, pq_tuple):
        assert len(pq_tuple) == 2, "len(pq_tuple) != 2 ..."
        return any(pq_tuple in passage_question_set for passage_question_set in self.exclude_dataset_sets)

    def in_myself(self, pq_tuple):
        assert len(pq_tuple) == 2, "len(pq_tuple) != 2 ..."
        return pq_tuple in self.passage_question_set

    def isdisjoint(self, passage_question_set):
        return self.passage_question_set.isdisjoint(passage_question_set)
