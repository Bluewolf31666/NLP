import os
import re
import pandas as pd
from tokenizers import Tokenizer
from nltk.tokenize import RegexpTokenizer
from konlpy.tag import Okt, Kkma

class UserTokenizers:
    """class info : 사용자의 필요에 맞게 다양한 토크나이저를 적용하는 모듈"""
    def __init__(self) -> None:
        self.bpe_tokenizer_pretrained= Tokenizer.from_file("./mode_data/bpe_tokenizer.json")
        self.wp_tokenizer_pretrained= Tokenizer.from_file("./mode_data/wp_tokenizer.josn")
        self.okt=Okt()
        self.kkma=Kkma()
    
    @staticmethod
    def whitespaceToken( data: str)-> list:
        """
        funcinfo whitespaceToken: 공백문자로 데이터를 나누는 토큰화
        param data : 토큰화할 문자 데이터
        return token_rs: 토큰 결과 
        """
        token_rs= data.split(' ')
        return token_rs
    @staticmethod
    def regexsplitToken(data: str, pat: str = '[\.\,!?\n]')-> list:
        """
        funcinfo regexsplitToken: 정규표현식의 패턴을 기준으로 데이터를 토큰화
        param data : 토큰화할 문자 데이터
        param pat : 토큰화할 정규식
        return token_rs: 토큰 결과
        """
        re_rs= re.split(pat, data, maxsplit=0)
        token_rs=[rs_unit.strip()for rs_unit in re_rs if len(rs_unit.strip())>1]
        return token_rs
    @staticmethod
    def regexselectToken(data: str, pat: str='[\w]+') -> list:
        """
        funcinfo regexselectToken: 정규표현식의 패턴을 선택하여 토큰화
        param data : 토큰화할 문자 데이터
        param pat : 토큰화할 정규식
        return token_rs: 토큰 결과
        """
        token_rs = RegexpTokenizer(pat).tokenize(data)
        return token_rs
    
    def BPETokenizer(self, data : str) -> list :
        """
        funcinfo BPETokenizer: char bpe로 훈련된 모델로 토큰화
        param data : 토큰화할 문자 데이터
        return token_rs: 토큰 결과
        """
        token_rs = self.bpe_tokenizer_pretrained.encode(data).tokens
        
        return token_rs
    def WordPieceTokenizer(self, data : str) -> list:
        """
        funcinfo WordPieceTokenizer: bert word piece로 훈련된 모델로 토큰화
        param data : 토큰화할 문자 데이터
        return token_rs: 토큰 결과
        """
        token_rs = self.wp_tokenizer_pretrained.encode(data).tokens
        
        return token_rs
    def konlyMorphsTokenizer(self, data: str)-> list:
        """
        funcinfo konlyMorphsTokenizer:Okt 형태소 분석 결과
        param data : 토큰화할 문자 데이터
        return token_rs: 토큰 결과        
        """
        token_rs= self.okt.morphs(data)
        return token_rs
    def konlpyNounsTokenizer(self, data: str) -> list:
        """
        funcinfo konlpyNounsTokenizer:Kkama 명사 분석 결과
        param data : 토큰화할 문자 데이터
        return token_rs: 토큰 결과  
        """
        token_rs=self.kkma.nouns(data)
        return token_rs