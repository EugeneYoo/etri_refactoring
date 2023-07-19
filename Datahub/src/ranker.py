import modelDir
import logging
from classes.config_phrase import args
from gensim.models import Doc2Vec
import re
import numpy as np
import pandas as pd
import torch
from QDS import QDS
# doc2vec test
import os

env = 'development' if not 'APP_ENV' in os.environ else os.environ['APP_ENV']
config = args[env]

logging.basicConfig(level=config['python_logging_lvl'])

class Ranker:
    instance = None
    def __init__(self):
        if self.instance is not None:
            raise ValueError("An instantiation already exists!")
        self.qds = QDS()
        self.qds.load_state_dict(torch.load(modelDir.modelDir["qds"]))
        self.qds.eval()

        try:
            self.qdsRawDataset = pd.read_csv(modelDir.datasetDir["qdsRawDataset"], encoding='utf-8-sig', index_col=False)
        except:
            self.qdsRawDataset = None
        self.doc2vec = Doc2Vec.load(modelDir.modelDir["doc2vec"])
        self.doc2vec.random.seed(9999)
        self.resultDatasetList = []
        self.queryVector = []
        self.clickedDatasetVector = []
        self.mode_modified_time= int(os.path.getmtime(modelDir.modelDir["qds"]))
        self.reloadQdsModel()

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = Ranker()
        return cls.instance

    def getResultDataset(self):
        return self.resultDatasetList

    def reloadQdsModel(self):
        logging.info("ranking model change")
        self.qds.load_state_dict(torch.load(modelDir.modelDir["qds"]))
        self.mode_modified_time = int(os.path.getmtime(modelDir.modelDir["qds"]))

    def getQueryVector(self):
        return self.queryVector

    def getClickedDatasetVector(self):
        return self.clickedDatasetVector

    def getQDSRawDataset(self):
        try:
            self.qdsRawDataset = pd.read_csv(r"C:\Users\hi\Desktop\khu_2\khu\Datahub\dataset\QDSRawDataset.csv", encoding='utf-8-sig', index_col=False)
        except:
            self.qdsRawDataset = None
        return self.qdsRawDataset

    def getClickedDatasetTitle(self):
        return self.clickedDatasetTitle

    def getQueryString(self):
        return self.queryString

    def setResultDataset(self, datasetList):
        self.resultDatasetList = datasetList

    ###
    def setQueryVector(self, query):
        query = query.split(' ')
        self.queryVector = self.doc2vec.infer_vector(query, steps=20, alpha=0.025)

    def setDatasetVector(self, datasetList):
        for elem in datasetList:
            logging.info("get for function")
            metadata = elem["title"] + " " + elem["desc"]
            #특수문자 제거
            line1 = re.sub('(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', metadata)
            line2 = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》\_]', ' ', line1)
            line3 = re.sub(re.compile(r'\s+'), ' ', line2)
            line4 = line3.strip()
            metadata = line4
            documentVector = self.doc2vec.infer_vector(metadata.split(' '), steps=20, alpha=0.025)
            elem["documentVector"] = documentVector

        return datasetList




    def _score(self, query, datasetList):
        '''
        :param query: 사용자 질의
        :param data: 결과 데이터셋리스트
        :return: query와 data의 임베딩 값 비교를 통한 유사도 결과 리스트 {'id' : 1, 'score' : 13}
        사용자 질의와 결과 데이터셋 리스트를 입력받아 질의와 각 데이터셋 별 score를 계산하고 이를 반환
        '''
        logging.debug(query)
        self.queryString = query
        query = query.split(' ')
        self.queryVector = self.doc2vec.infer_vector(query, steps=20, alpha=0.025)
        logging.info("query infer complete")
        scoreList = []
        for elem in datasetList:
            logging.info("get for function")
            metadata = elem["title"] + " " + elem["desc"]
            #특수문자 제거
            line1 = re.sub('(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', metadata)
            line2 = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》\_]', ' ', line1)
            line3 = re.sub(re.compile(r'\s+'), ' ', line2)
            line4 = line3.strip()
            metadata = line4
            try:
                documentVector = self.doc2vec.infer_vector(metadata.split(' '), steps=20, alpha=0.025)
                elem["documentVector"] = documentVector
                score = self.qds(torch.FloatTensor(np.concatenate((self.queryVector,documentVector), axis=0)))

                scoreList.append({'uid' : elem["uid"], 'qdsScore' : score.item(), 'elasticScore' : elem["_score"]})
            except:
                score = 0
                scoreList.append({'uid' : elem["uid"], 'qdsScore' : score, 'elasticScore' : elem["_score"]})
        logging.info("scoring finish")
        return scoreList, datasetList



    def rank(self, datasetList, query):
        '''
        :param dataDict: {id : datasetInfo} 형태의 딕셔너리
        :param query: 사용자 질의
        dataDict와 query의 score 계산 순서대로 list에 {id : 1, score : 90} 객체 append
        그 후 score 기준으로 sorting
        :return: score 순으로 정렬된 dataInfo 리스트
        '''
        logging.info("in rank function")
        scoreList, datasetList = self._score(query, datasetList)
        #qds Score 높은 순으로 정렬
        scoreList = sorted(scoreList, key=lambda document: (document["qdsScore"] + document["elasticScore"] * modelDir.params["scoreWeight"]), reverse=True)

        return scoreList, datasetList

    def negativeSampling(self, datasetId):
        candidateVectors = []

        for dataset in self.resultDatasetList:
            if dataset["uid"] == int(datasetId):
                self.clickedDatasetVector = dataset["documentVector"]

                self.clickedDatasetTitle = dataset["title"]
            else:
                candidateVectors.append(dataset["documentVector"])

        if len(candidateVectors) >= 2:
            # 최대 2개 false data
            negatives = np.random.permutation(candidateVectors)[:2]
        elif len(candidateVectors) == 1:
            negatives = [candidateVectors[0]]
        else:
            negatives = []

        return negatives