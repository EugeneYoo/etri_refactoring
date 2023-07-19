import os
import logging
from .config_phrase import args
from gensim.models import KeyedVectors

env = 'development' if not 'APP_ENV' in os.environ else os.environ['APP_ENV']
config = args[env]

logging.basicConfig(level=config['python_logging_lvl'])

class QueryIntentAnalyzer:
    def __init__(self, model):
        """
        사용자 의도 분석 클래스

        Params
        ------
        pretrained_model: 도메인 특화 모델
        model_path: 모델 경로
        model_modi_date: 모델 파일 작성 시간

        :param model: 불러올 pre-trained 사용자 의도 분석 모델
        """
        self.pretrained_model = KeyedVectors.load_word2vec_format(model, binary=True, unicode_errors='ignore')
        self.model_path = model
        self.model_modi_time = int(os.path.getmtime(model))


    def modi_check_and_reload_model(self):
        """
        모델 수정이 감지되면 모델을 다시 로딩하고 True를 반환
        아니라면 False 반환
        """
        if self.model_modi_time != int(os.path.getmtime(self.model_path)):
            logging.debug(f"모델 변경: {self.model_modi_time} => {int(os.path.getmtime(self.model_path))}")
            self.pretrained_model = KeyedVectors.load_word2vec_format(self.model_path, binary=True, unicode_errors='ignore')
            self.model_modi_time = int(os.path.getmtime(self.model_path))
            return True
        return False

    def query_intent_analysis(self, query, threshold=0.5):
        """
        사용자 의도 분석을 수행하고, threshold를 넘기는 결과를 반환함

        :param query: (string) 사용자 질의
        :param threshold: (float) 사용자 의도 분석 결과 threshold, 점수 미달 결과 제거
        :return: 사용자 의도 분석 결과
        """
        split_query = query.split()

        result_dic = {}
        intent_value = []

        # query 분절
        for query in split_query:
            try:
                # 의도 분석 모델에 query에 해당하는 분석 결과 받아옴 top 5개
                intents = self.pretrained_model.wv.most_similar(query, topn=config['expand_limit'])
                temp_list = []
                for intent in intents:
                    # 결과중 threshold를 넘기는 값을 결과 값에 append
                    if intent[1] >= threshold:
                        temp_list.append(intent[0])
                # 마지막으로 사용자 query를 포함시킴.
                data = query
                temp_list.append(data)
                temp_list[0], temp_list[-1] = temp_list[-1], temp_list[0]
                intent_value.append(temp_list)

            # 의도 분석 결과가 나오지 않은 경우 qeury만 추가함.
            except:
                temp_list = []
                data = query
                temp_list.append(data)

                intent_value.append(temp_list)

        for i in range(0, len(split_query)):
            result_dic.update({split_query[i]: intent_value[i]})

        return result_dic

    def query_intent_analysis_with_score(self, query, threshold=0.5):
        """
        사용자 의도 분석을 수행하고, threshold를 넘기는 결과를 반환함(점수 포함)

        :param query: (string) 사용자 질의
        :param threshold: (float) 사용자 의도 분석 결과 threshold, 점수 미달 결과 제거
        :return: 사용자 의도 분석 결과(점수 포함)
        """
        #택지 정보 [택지,정보]
        split_query = query.split()

        result_dic = {}
        intent_value = []
        # query 분절
        for query in split_query:
            try:
                # 의도 분석 모델에 query에 해당하는 분석 결과 받아옴 top 5개
                intents = self.pretrained_model.wv.most_similar(query, topn=config['expand_limit'])
                temp_list = []
                for intent in intents:
                    temp_dic ={}
                    # 결과중 threshold를 넘기는 값을 결과 값에 append
                    if intent[1] >= threshold:
                        temp_dic['intent'] = intent[0]
                        temp_dic['score'] = intent[1]
                        temp_list.append(temp_dic)
                '''
                # 마지막으로 사용자 query를 포함시킴. 1.0의 score 부여.
                data = (query, 1.0)
                temp_list.append(data)
                '''
                intent_value.append(temp_list)

            # 의도 분석 결과가 나오지 않은 경우 빈 리스트 추가
            except:
                temp_list = []
                '''
                data = (query, 1.0)
                temp_list.append(data)
                '''
                intent_value.append(temp_list)
        '''
        for i in range(0, len(split_query)):
            result_dic.update({split_query[i]: intent_value[i]})
        '''
        result_list = []
        for i in range(0, len(split_query)):
            temp = {}
            temp['keyword'] = split_query[i]
            temp['intents'] = intent_value[i]
            result_list.append(temp)

        return result_list

    def query_intent_analysis_with_score_query(self, query, size=config['expand_limit'], threshold=0.5):
        """
        사용자 의도 분석을 수행하고, threshold를 넘기는 결과를 반환함(점수 포함)

        :param query: (string) 사용자 질의
        :param threshold: (float) 사용자 의도 분석 결과 threshold, 점수 미달 결과 제거
        :return: 사용자 의도 분석 결과(점수 포함)
        """
        #택지 정보 [택지,정보]
        split_query = query.split()

        result_dic = {}
        intent_value = []
        # query 분절
        for query in split_query:
            try:
                # 의도 분석 모델에 query에 해당하는 분석 결과 받아옴 top 5개
                querybin = KeyedVectors.load_word2vec_format(r'C:\Users\hi\Desktop\khu_2\khu\Datahub\qeury_txt.bin', binary=True, unicode_errors='ignore')
                intents = querybin.wv.most_similar(query, topn=size)
                temp_list = []
                for intent in intents:
                    temp_dic ={}
                    # 결과중 threshold를 넘기는 값을 결과 값에 append
                    if intent[1] >= threshold:
                        temp_dic['intent'] = intent[0]
                        temp_dic['score'] = intent[1]
                        temp_list.append(temp_dic)
                '''
                # 마지막으로 사용자 query를 포함시킴. 1.0의 score 부여.
                data = (query, 1.0)
                temp_list.append(data)
                '''
                intent_value.append(temp_list)

            # 의도 분석 결과가 나오지 않은 경우 빈 리스트 추가
            except:
                temp_list = []
                '''
                data = (query, 1.0)
                temp_list.append(data)
                '''
                intent_value.append(temp_list)
        '''
        for i in range(0, len(split_query)):
            result_dic.update({split_query[i]: intent_value[i]})
        '''
        result_list = []
        for i in range(0, len(split_query)):
            temp = {}
            temp['keyword'] = split_query[i]
            temp['intents'] = intent_value[i]
            result_list.append(temp)

        return result_list
