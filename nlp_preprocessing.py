import pandas as pd
from konlpy.tag import Okt, Kkma
from Token import UserTokenizers
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TrainTransfromVect:
    def __init__(self):

        self.ut_cls=UserTokenizers()

    def fitWP_TFIDF(self):
        """
        info : 훈련된 워드피스 토크나이저로 tfidf 벡터화 모델선언
        """        
        self.wp_tfidf= TfidfVectorizer(tokenizer=self.ut_cls.WordPieceTokenizer)
        return self.wp_tfidf   
    def fitBPE_TFIDF(self):
        """
        info : 훈련된 BPE 토크나이저로 tfidf 백터화 모델선언
        주의 : 토크나이저 훈련 모델에서 <unk>문제 해결 필요
        """
        self.bpe_tfidf = TfidfVectorizer(tokenizer= self.ut_cls.BPETokenizer)
        return self.bpe_tfidf
    def fitKKMA_TFIDF(self):
        """
        info : 훈련된 Konlpy.Kkma 토크나이저로 tfidf 백터화 모델선언
        사용주의: 메모리에러, 소요시간 주의
        """
        self.kkma_tfidf = TfidfVectorizer(tokenizer=self.ut_cls.konlpyNounsTokenizer)
        
        return self.kkma_tfidf
    def fitMP_TFIDF(self):
        """
        info : 훈련된 konlpy.Okt 토크나이저로 tfidf 벡터화 모델선언
        사용주의: 메모리에러, 소요시간 주의
        """
        self.mp_tfidf= TfidfVectorizer(tokenizer=self.ut_cls.konlyMorphsTokenizer)
        
        return self.mp_tfidf
    
    def fit_run(self, user_token_nm,data) :
        """
        info : tfidf 벡터화 수행 
        param user_token_nm : 토크나이저 선택 {'wp': '워드피스','bpe':'BPE','kkma':'꼬꼬마','mp':'Okt'}
        param data:백터 모델 훈련 데이터
        """ 
        if user_token_nm =='wp':
            self.vec_model= self.fitWP_TFIDF()
            # 모델 저장 코드 추가 필요
            
        elif user_token_nm =='bpe':
            self.vec_model= self.fitBPE_TFIDF()
            # 모델 저장 코드 추가 필요
        elif user_token_nm =='kkma':
            self.vec_model= self.fitKKMA_TFIDF()
            # 모델 저장 코드 추가 필요
        elif user_token_nm =='mp':
            self.vec_model= self.fitMP_TFIDF()
            # 모델 저장 코드 추가 필요
            
        else :
            # 모델 개발 할 때 혹은 인수인계자에게 디버깅하기 좋은 코드로 넘길 때 사용
            raise ValueError("user_token_nm이 올바르지 않습니다.['wp','bpe','kkma','mp'] ")
            # 서비스 운영을 할 때 기본설정하고, logging을 이용해서 로그를 남기거나 슬랙 알람/ 메일링 주는 모듈이 실행되게 함
            # self.vec_model = self.fitWP_TFIDF()
        self.vec_model.fit(data)
        ## 저장된 모델이 있는 경우 : 모델을 불러오는 모듈로 전환
    
    def transform_run(self, data, chunk_size):
        """
        info: tfidf 백터화 하여 np.array로 변환
        param data : 벡터로 변환하려는 데이터
        param chunk_size : np.array로 변환하는 단위, 데이터 수
        return vec_arr: np.array 
        """
        ## 데이터 수를 조정해서 데이터를 변환함
        data_len= len(data)
        for st_idx in tqdm(range(0, data_len, chunk_size)):
            tmp_data = data[st_idx:st_idx+chunk_size]
            
            if st_idx==0:
                vac_arr= self.vec_model.transform(tmp_data).toarray()
            else :
                tmp_data_arr= self.vec_model.transform(tmp_data).toarray()
                vac_arr= np.append(vac_arr,tmp_data_arr, axis=0)
        return vac_arr