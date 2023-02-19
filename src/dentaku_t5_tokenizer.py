from typing import List
import re

from transformers import T5Tokenizer

__all__ = ["DentakuT5Tokenizer"]

class DentakuT5Tokenizer(T5Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        
    def _tokenize(self, text, **kwargs) -> List[str]:
        token_list = super()._tokenize(text, **kwargs)
        digit_awared_token_list = []
        
        for token in token_list:
            if re.match(r"(\d)+", token) or re.match(r"▁(\d)+", token) or re.match(r"\-(\d)+", token) or re.match(r"▁\-(\d)+", token):
                # 2桁以上の数字がまとめてsubwordになっている時
                digit_awared_token_list.extend(token)
                
            elif re.match(r"▁[+\-*%&>=/;]", token):
                # 演算子も必ず　"▁"　+ "{演算子}"に分ける
                digit_awared_token_list.extend(token)
            else:
                digit_awared_token_list.append(token)

        return digit_awared_token_list
