from pathlib import Path
import pickle

__all__ = ["PickleFileLoader", "PikleFileWriter"]

class PickleFileLoader:
    def __init__(self, file_path:Path):
        self.file_path = Path(file_path)
        
    def __iter__(self):
        with self.file_path.open(mode="rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break


class PickleFileWriter:
    def __init__(self, file_path:Path):
        self.file_path = Path(file_path)
        self.file_obj = None
        
    def __enter__(self):
        self.file_obj = self.file_path.open(mode="wb")

    def __exit__(self):
        self.close()

    def write(self, obj):
        pickle.dump(obj, self.file_obj)
        
    def close(self):
        self.file_obj.close()
