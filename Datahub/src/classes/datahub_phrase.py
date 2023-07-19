import logging
import os
from .config_phrase import queries

from konlpy.tag import Okt, Mecab
Mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")  # Mecab 형태소 분석기 로딩
from itertools import product

from elasticsearch import Elasticsearch

from .queryintentanalyzer_phrase import QueryIntentAnalyzer

from .config_phrase import args
env = 'development' if not 'APP_ENV' in os.environ else os.environ['APP_ENV']
config = args[env]

logging.basicConfig(level=config['python_logging_lvl'])

# desc: 시맨틱 서치 포털 클래스
# input
# self.hub_index_name: 데이터 허브의 데이터셋이 저장되어 있는 elasticsearch index name
# self.es : python을 통해 elasticsearch를 사용하기 위한 클라이언트 객체
# self.intent_analyzer: 사용자 의도 분석을 수행하는 QueryIntentAnalyzer 객체, 도메인 특화 모델을 사용해 초기화 함
class DataHub:
    """Datahub 시맨틱 서치 허브 클래스

     :param self.hub_index_name: 데이터 허브의 데이터셋이 저장되어 있는 elasticsearch index name
     :param self.es : python을 통해 elasticsearch를 사용하기 위한 클라이언트 객체
     :param self.intent_analyzer: 사용자 의도 분석을 수행하는 QueryIntentAnalyzer 객체, 도메인 특화 모델을 사용해 초기화 함
    """

    def __init__(self, **kwargs):
        """
         :param

         model: word2vec bin 파일

         elasticsearch: hub dataset이 저장된 elasticsearch 서버 주소

         hub_index_name: hub의 dataset이 저장된 es index name
        """
        self.hub_index_name = 'localhost5500' if 'hub_index_name' not in kwargs else kwargs['hub_index_name']
        use_ssl = True if 'use_ssl' not in kwargs else kwargs['use_ssl']
        verify_certs = False if 'verify_certs' not in kwargs else kwargs['verify_certs']

        # logging.debug(kwargs['verify_certs'], verify_certs)

        elasticsearch_host = 'localhost:9200' if 'elasticsearch' not in kwargs else kwargs['elasticsearch']
        self.es = Elasticsearch(elasticsearch_host, use_ssl=use_ssl, verify_certs=verify_certs)

        # 포털이 사용할 모델 binary file을 넣어서 analyzer 객체 초기화
        self.intent_analyzer = QueryIntentAnalyzer(
            './model/ko_new.bin') if 'model' not in kwargs else QueryIntentAnalyzer(kwargs['model'])

    def make_new_analyzer_object(self, model):
        """

        Args:
            model: 모델 위치

        Returns:

        """
        self.intent_analyzer = QueryIntentAnalyzer(model)

    def search(self, query_list):
        """elastic search form의 질의를 받아 검색을 수행하는 함수

        :param query_list: 검색을 수행할 elastic search form query
        :return: 검색 결과 반환 (list of dicts)
        """

        dataset_list = []
        ex_dataset_list = []
        basic_dataset_list = []
        extended_dataset_list = []

        if len(query_list) == 0:
            return []

        # check query_list
        logging.debug(query_list)
        query_list = str(query_list)
        
        # classes의 queries.py에서 불러와서 쿼리 생성
        query_body = queries['semantic_search_query'].copy()
        query_body['query']['query_string']['query']= query_list

        # 검색 결과 받아오는 사이즈 크기
        size = 10000
        logging.debug(f"hub index name:{self.hub_index_name}")
        result = self.es.search(index=self.hub_index_name, body=query_body, size=size)
       
        search_result = []
        #result = test_result
        logging.debug(f"spend time using es python api: {result['took']} , {result['hits']['total']['value']}")
        total_size = result['hits']["total"]["value"]
        # 검색 결과가 존재하는 경우
        if total_size != 0:
            # 만약 size가 전체 검색 결과 미만인 경우
            if size < total_size:
                while True:
                    # 현재 검색 결과 리스트에 저장
                    for data in result['hits']['hits']:
                        dataset = data['_source']
                        dataset['_score'] = data['_score']
                        search_result.append(dataset)
                    dataset_list.extend(search_result)

                    # 마지막 위치를 나타내는 sort 값
                    sort_value = data['sort']

                    # 기존 쿼리에 search_after 적용
                    result_body['search_after'] = sort_value
                    # 해당 위치부터 재검색 수행
                    result = self.es.search(index=self.hub_index_name, body=result_body, size=size)
                    # 다음 결과가 없는 경우 break
                    if len(result['hits']['hits']) == 0:
                        break
            # 만약 size 가 전체 검색 결과 이상인 경우 (전체 검색 결과를 한 번에 받아 온 경우)
            elif size >= total_size:
                for data in result['hits']['hits']:
                    dataset = data['_source']
                    dataset['_score'] = data['_score']
                    search_result.append(dataset)
                dataset_list.extend(search_result)

        if (len(dataset_list) > 0):
            #logging.debug("hits!!")
            basic_dataset_list = list({data['uid']: data for data in dataset_list}.values())
            '''
            return_list = list(
                map(dict, collections.OrderedDict.fromkeys(tuple(sorted(dataset.items())) for dataset in dataset_list)))
            '''
        else:
            basic_dataset_list = dataset_list
        '''
        # extended query
        ex_query_body = queries['semantic_search_query_extended_metadata'].copy()
        ex_query_body['query']['nested']['query']]['query_string']['query']= query_list

        logging.debug(result_body)

        # 검색 결과 받아오는 사이즈 크기
        size = 10000
        result = self.es.search(index=self.hub_index_name, body=result_body, size=size)
        search_result = []

        total_size = result['hits']["total"]["value"]
        # 검색 결과가 존재하는 경우
        if total_size != 0:
            # 만약 size가 전체 검색 결과 미만인 경우
            if size < total_size:
                while True:
                    # 현재 검색 결과 리스트에 저장
                    for data in result['hits']['hits']:
                        dataset = data['_source']
                        search_result.append(dataset)
                    ex_dataset_list.extend(search_result)

                    # 마지막 위치를 나타내는 sort 값
                    sort_value = data['sort']

                    # 기존 쿼리에 search_after 적용
                    result_body['search_after'] = sort_value
                    # 해당 위치부터 재검색 수행
                    result = self.es.search(index=self.hub_index_name, body=result_body, size=size)
                    # 다음 결과가 없는 경우 break
                    if len(result['hits']['hits']) == 0:
                        break
            # 만약 size 가 전체 검색 결과 이상인 경우 (전체 검색 결과를 한 번에 받아 온 경우)
            elif size >= total_size:
                for data in result['hits']['hits']:
                    dataset = data['_source']
                    search_result.append(dataset)
                ex_dataset_list.extend(search_result)

        if (len(ex_dataset_list) > 0):
            logging.debug("hits!!")
            extended_dataset_list = list({data['uid']: data for data in dataset_list}.values())
            """
            return_list = list(
                map(dict, collections.OrderedDict.fromkeys(tuple(sorted(dataset.items())) for dataset in dataset_list)))
            """
        else:
            extended_dataset_list = dataset_list
        '''

        extended_dataset_list = []

        # 두 리스트 병합
        merge_list = basic_dataset_list + extended_dataset_list

        result_list = []
        # removing the duplicate entry
        for i in range(len(merge_list)):
            if merge_list[i] not in merge_list[i + 1:]:
                result_list.append(merge_list[i])

        #logging.debug("return final list")
        # return merge_list
        return basic_dataset_list

    def analyze_intent(self, query, similarity=0.5):
        """self.intent_analyzer로 사용자 의도 분석 요청

        :param query:(string) 분석할 사용자 질의
        :param similarity:(float) 사용자 의도 분석 결과 similarity

        :return: 데이터 허브에서 분석한 사용자 의도 분석 결과 반환 ex) query: "재정 데이터" =>{"재정": [["세입"], ["재정"]], "데이터": [["정보"], ["데이터베이스"], ["데이터"]]}
        """
        intent = self.intent_analyzer.query_intent_analysis(query, similarity)
        return intent

    def analyze_intent_with_score(self, query, similarity=0.5):
        """self.intent_analyzer로 사용자 의도 분석 요청, 결과 값에 분석 점수를 포함하여 받음.

        :param query:(string) 분석할 사용자 질의
        :param similarity:(float) 사용자 의도 분석 결과 similarity
        :return: 데이터 허브에서 분석한 사용자 의도 분석 결과 반환 ex) query: "재정 데이터" =>{"재정": [["세입",0.62], ["재정", 1.0]], "데이터": [["정보", 0.53], ["데이터베이스", 0.51], ["데이터", 1.0]]}
        """
        intent = self.intent_analyzer.query_intent_analysis_with_score(query, similarity)
        return intent

    def load_new_model(self, model_name):
        """
        해당하는 모델 이름으로 intent_analyzer 초기화

        :param model_name: (string) 새로 불러올 모델 이름
        :return: 새로운 모델을 불러온 QueryIntentAnalyzer 객체
        """
        self.intent_analyzer = QueryIntentAnalyzer(model_name)

    # desc: 의도 분석 결과로 쿼리 생성을 위한 함수
    # input
    # intents: 의도 분석 결과
    # curr_list: 의도 분석 결과 저장
    # output: 생성된 쿼리가 list in list 형태로 담김
    def make_query(self, intents, curr_list=[[]]):
        """의도 분석 결과로 쿼리 생성을 위한 함수

        :param intents: 의도 분석 결과 {"A": ["A", "C"], "B": ["B", "D"]}
        :param curr_list: 생성된 단어 묶음을 저장하기 위한 공간
        :return: 생성된 쿼리 묶음. [["A","B"],["A","D"],["C","B"], ["C","D"]]
        """
        #logging.debug(intents)

        if len(intents) == 0:
            return curr_list

        keyword = list(intents.keys())[0]

        next_list = []
        for item in curr_list:
            for new_item in intents[keyword]:
                next_list.append(item + [new_item])

        del intents[keyword]
        return self.make_query(intents, next_list)

    def make_query_style(self, query_list):
        """
        이중리스트에 담긴 쿼리를 elasticsearch에 알맞게 변환함.\n
        str list sub element join with and, str list element join with or

        :param query_list: make_query로 만들어진 list ln list
        :return: elastic search qeury form [['A', 'B'], ['A', 'C']] => ((A AND B) OR (A AND C))
        """
        modified_query_list = []
        for query in query_list:
            tmp_query = []
            for elem in query:
                # 명사구인 경우
                if "_" in elem:
                    elem = elem.split('_')
                    elem = f'\"{" ".join(elem)}\"'
                tmp_query.append(elem)
            sub_body = " AND ".join(tmp_query)
            modified_query_list.append("(" + sub_body + ")")
        body = " OR ".join(modified_query_list)
        return body

    def semantic_search(self, query, algorithm_id="", algorithm_parameter=0.5):
        """
        단일 포털 검색 시 수행하는 semantic search

        :param query: (string) 사용자 질의
        :param algorithm_id: (string) 사용할 algorithm_id 1: 사용자 threshold 적용, else: 기본 값 적용
        :param algorithm_parameter: (float) threshold 값.
        :return: (list in dict) qeury를 통해 검색한 데이터셋 리스트
        """

        if algorithm_id is '1':
            # threshold 적용'
            phrase = self.check_query_phrase(query)
            #logging.debug(f"phrase::{phrase}")
            phrase = " ".join(phrase)
            #logging.debug(f"phrase::{phrase}")
            intents = self.analyze_intent(phrase, algorithm_parameter)
        else:
            intents = self.analyze_intent(query)

        intent_query = self.make_query(intents)
        query = self.make_query_style(intent_query)
        result = self.search(query)
        return result

    def re_search(self, intent_list):
        """
        사용자가 선택한 키워드로 elastic search form의 질의를 생성해 재검색을 수행하는 함수

        :param intent_list: 재검색을 수행할 키워드
        :return: 검색 결과 반환 (list of dicts)
        """

        facet_list = []
        dataset_list = []
        ex_dataset_list = []
        basic_dataset_list = []
        extended_dataset_list = []

        if len(intent_list) == 0:
            return []

        for selected_keywords in intent_list:
            # 키값을 우선 추가
            keyword = selected_keywords['keyword']
            value = selected_keywords['intents']
            temp = []
            if keyword not in value:
                temp.append(keyword)
            temp.extend(value)
            facet_list.append(temp)

        query_dict = {}
        query_dict["query"] = list(product(*facet_list))

        logging.debug(query_dict)
        # str list sub element join with and, str list element join with or
        modified_query_list = []
        for query in query_dict["query"]:
            tmp_query = []
            for elem in query:
                # 명사구인 경우
                if "_" in elem:
                    elem = elem.split('_')
                    elem = f'\"{" ".join(elem)}\"'
                tmp_query.append(elem)
            sub_body = " AND ".join(tmp_query)
            modified_query_list.append("(" + sub_body + ")")
        query_list = " OR ".join(modified_query_list)
        logging.debug("re-search Keyword:", body)
        
        # classes의 queries.py에서 불러와서 쿼리 생성
        query_body = queries['semantic_search_query'].copy()
        query_body['query']['query_string']['query']= query_list


        result = self.es.search(index=self.hub_index_name, body=query_body, size=1000)

        if result['hits']["total"]["value"] != 0:
            search_result = []
            for data in result['hits']['hits']:
                dataset = data['_source']
                search_result.append(dataset)
            dataset_list.extend(search_result)

        basic_dataset_list = list({data['uid']: data for data in dataset_list}.values())
        """
        # 추가된 es body
        query_body = queries['semantic_search_query_extended_metadata'].copy()
        query_body['query']['query_string']['query']= query_list

        result = self.es.search(index=self.hub_index_name, body=result_body, size=1000)

        if result['hits']["total"]["value"] != 0:
            search_result = []
            for data in result['hits']['hits']:
                dataset = data['_source']
                search_result.append(dataset)
            ex_dataset_list.extend(search_result)

        extended_dataset_list = list({data['uid']: data for data in ex_dataset_list}.values())

        # 두 리스트 병합
        merge_list = basic_dataset_list + extended_dataset_list

        result_list = []
        # removing the duplicate entry
        for i in range(len(merge_list)):
            if merge_list[i] not in merge_list[i + 1:]:
                result_list.append(merge_list[i])

        logging.debug("return final list")

        """
        return basic_dataset_list

    def get_search_aggs(self, query, algorithm_id="", algorithm_parameter=0.5):
        if algorithm_id is '1':
            # threshold 적용
            intents = self.analyze_intent(query, algorithm_parameter)
        else:
            intents = self.analyze_intent(query)

        intent_query = self.make_query(intents)
        query = self.make_query_style(intent_query)

        # classes의 queries.py에서 불러와서 쿼리 생성
        query_body = queries['aggregation_query'].copy()
        query_body['query']['query_string']['query']= query

        result = self.es.search(index=self.hub_index_name, body=query_body, size=0)

        agg = {}
        if len(result['aggregations']["keywords"]["buckets"]) != 0:
            publisher_result = []
            for data in result['aggregations']["keywords"]["buckets"]:
                temp = {}
                temp['key'] = data['key']
                temp['doc_count'] = data['doc_count']
                publisher_result.append(temp)
            agg['publisher'] = publisher_result

        if len(result['aggregations']["dates"]["buckets"]) != 0:
            date_result = []
            for data in result['aggregations']["dates"]["buckets"]:
                temp = {}
                temp['key'] = data['key_as_string']
                temp['doc_count'] = data['doc_count']
                date_result.append(temp)
            agg['dates'] = date_result

        return agg

    def get_re_search_aggs(self, intent_list):
        facet_list = []
        dataset_list = []

        if len(intent_list) == 0:
            return []

        for selected_keywords in intent_list:
            # 키값을 우선 추가
            keyword = selected_keywords['keyword']
            value = selected_keywords['intents']
            temp = []
            if keyword not in value:
                temp.append(keyword)
            temp.extend(value)
            facet_list.append(temp)

        query_dict = {}
        query_dict["query"] = list(product(*facet_list))

        logging.debug(query_dict)
        # str list sub element join with and, str list element join with or
        modified_query_list = []
        for elem in query_dict["query"]:
            sub_body = " AND ".join(elem)
            modified_query_list.append("(" + sub_body + ")")

        query = " OR ".join(modified_query_list)
        logging.debug("re-search Keyword:", query)

        # classes의 queries.py에서 불러와서 쿼리 생성
        query_body = queries['aggregation_query'].copy()
        query_body['query']['query_string']['query']= query

        result = self.es.search(index=self.hub_index_name, body=query_body, size=0)

        agg = {}
        if len(result['aggregations']["keywords"]["buckets"]) != 0:
            publisher_result = []
            for data in result['aggregations']["keywords"]["buckets"]:
                temp = {}
                temp['key'] = data['key']
                temp['doc_count'] = data['doc_count']
                publisher_result.append(temp)
            agg['publisher'] = publisher_result

        if len(result['aggregations']["dates"]["buckets"]) != 0:
            date_result = []
            for data in result['aggregations']["dates"]["buckets"]:
                temp = {}
                temp['key'] = data['key_as_string']
                temp['doc_count'] = data['doc_count']
                date_result.append(temp)
            agg['dates'] = date_result

        return agg


    def check_query_phrase(self, query):
        """
        사용자가 입력한 쿼리를 입력 받아, 최대 길이의 명사구를 반환함.
        :param query: 사용자 입력 쿼리
        :return: 질의에서 확인되는 최대 길이 명사구
        """
        # 공백 단위로 질의 분절
        queryList = query.split()
        # 명사구 및 명사 찾아서 저장하는 리스트
        resultList = []

        # 공백 단위로 분절된 단어 리스트에서 인접한 단어를 1개 씩 이은 후 해당 단어 조합이
        # 사용하는 모델에서 명사구로 인식이 되는지에 따라 명사구 및 명사로 저장
        i = 0
        while(i < len(queryList)):
            #logging.debug("hi")
            tempIndex = None
            phraseCandidate = [queryList[i]]
            temp = ""
            # 단어 리스트를 iteration하여 1개씩 단어들을 이어붙이면서 명사구가 될 수 있는지 확인
            for j in range(i + 1, len(queryList)):
                phraseCandidate.append(queryList[j])
                # 명사구는 '_' 기호로 어간이 묶여서 저장되어 있음, 각 단어의 사이를 해당 기호로 연결하여 확장 여부 확인
                # 확장되는 단어가 하나라도 존재하는 경우
                checkword = "_".join(phraseCandidate)
                phrase_test = self.analyze_intent(checkword) # 모델을 사용한 확장 시도
                logging.debug(f'detected phrase: {phrase_test})')
                if len(phrase_test[checkword]) > 1:
                    temp = "_".join(phraseCandidate)
                    tempIndex = j
                if j == len(queryList) - 1 and temp == "":
                    resultList.append(queryList[i])
                elif j == len(queryList) - 1 and temp != "":
                    resultList.append(temp)
            if i == len(queryList) - 1:
                resultList.append(queryList[i])
            if tempIndex is not None:
                i = tempIndex + 1
            else:
                i += 1
        returnResult = []
        for index in range(len(resultList)):
            if "_" in resultList[index]:
                returnResult.append(resultList[index])
            else:
                returnResult.extend(Mecab.nouns(resultList[index]))

        #logging.debug("return result", returnResult)
        return returnResult






