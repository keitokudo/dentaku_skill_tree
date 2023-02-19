from typing import List
import re

from transformers import BartTokenizer

__all__ = ["DentakuBartTokenizer"]

class DentakuBartTokenizer(BartTokenizer):
    def _tokenize(self, text, **kwargs) -> List[str]:
        token_list = super()._tokenize(text, **kwargs)
        digit_awared_token_list = []
        
        for token in token_list:
            if re.match(r"(\d)+", token) or re.match(r"Ġ(\d)+", token) or re.match(r"\-(\d)+", token) or re.match(r"Ġ\-(\d)+", token):
                # 2桁以上の数字がまとめてsubwordになっている時
                digit_awared_token_list.extend(token)
                
            elif re.match(r"Ġ[+\-*%&>=/;^]", token) or re.match(r"Ġ[a-zA-Z]", token):
                # 演算子・変数名も必ず　"▁"　+ "{演算子}"に分ける
                digit_awared_token_list.extend(token)
            else:
                digit_awared_token_list.append(token)

        return digit_awared_token_list
