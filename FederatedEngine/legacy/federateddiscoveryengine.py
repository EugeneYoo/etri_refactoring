import json
import os
from itertools import product

import requests

from .config import args

from .datahubinfo import DataHubInfo

from gensim.models import KeyedVectors

# 동시 요청을 보내기 위한 라이브러리
from concurrent.futures import ThreadPoolExecutor

import psycopg2

# 환경변수를 불러와 config 세팅
# development, production 두 가지가 있음.
if 'APP_ENV' in os.environ:
    print(os.environ['APP_ENV'])

env = 'development' if not 'APP_ENV' in os.environ else os.environ['APP_ENV']
config = args[env]


def post_url(args):
    """
    동시 POST 요청을 위한 함수

    :param args: (post 요청할 urls, body)
    """
    try:
        result = requests.post(args[0], data=args[1], timeout=10)
        print(type(result))
        return result
    except requests.exceptions.Timeout as errd:
        print("Timeout Error : ", errd)
        print("URLs : ", args[0])
        print("ERRD:", type(errd))
        return errd

    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting : ", errc)
        print("URLs : ", args[0])
        print("ERRC:", type(errc))
        return errc

    except requests.exceptions.HTTPError as errb:
        print("Http Error : ", errb)
        print("URLs : ", args[0])
        print("ERRB:", type(errb))
        return errb

    # Any Error except upper exception
    except requests.exceptions.RequestException as erra:
        print("AnyException : ", erra)
        print("URLs : ", args[0])
        print("ERRA:", type(erra))
        return erra


class FederatedDiscoveryEngine:
    """
    통합 검색 엔진 클래스

    params
    ------
    integrate_model: 재분석을 위한 통합 및 재분석 전용 모델\n
    db : postgreSQL 연결 객체\n
    cursor : postgreSQL 질의 커서\n
    hub_list: 통합 검색이 수행가능한 데이터 허브 리스트\n

    """

    def __init__(self):
        self.integrate_model = KeyedVectors.load_word2vec_format(config['integrate_model'], binary=True,
                                                                unicode_errors='ignore')
        # user: 로그인/그룹 롤에서 본인의 계정, dbname: db name. 테이블 네임은 query에서 선택
        self.db = psycopg2.connect(host=config['pgsql_host'], dbname=config['pgsql_dbname'], user=config['pgsql_user'], \
                                    password=config['pgsql_password'], port=config['pgsql_port'],\
                                    keepalives=1, keepalives_idle=30, keepalives_interval=10,
                                    keepalives_count=5)
        self.cursor = self.db.cursor()

        self.hub_list = self.get_hub_list()

    def __del__(self):
        self.db.close()
        self.cursor.close()

    def make_query(self, intents, curr_list=[[]]):
        """의도 분석 결과로 쿼리 생성을 위한 함수

        :param intents: 의도 분석 결과 {"A": [["A"], ["C"]], "B": [["B"], ["D"]]}
        :param curr_list: 생성된 단어 묶음을 저장하기 위한 공간
        :return: 생성된 쿼리 묶음. [["A","B"],["A","D"],["C","B"], ["C","D"]]
        """
        if len(intents) == 0:
            return curr_list

        keyword = list(intents.keys())[0]

        next_list = []
        for item in curr_list:
            for new_item in intents[keyword]:
                next_list.append(item + [new_item])

        del intents[keyword]
        return self.make_query(intents, next_list)

    def get_hub_list(self):
        """
        통합 검색이 가능한 데이터 허브들을 불러옴 __init__에서 사용
        :return: (list(DataHubInfo))통합 검색이 가능한 데이터 허브 리스트
        """
        self.cursor.execute("select * from hub_info")
        result = self.cursor.fetchall()

        hub_list = []

        for data in result:
            hub = DataHubInfo()
            hub.set_hub_url(data[2])
            hub.set_hub_name(data[1])
            hub.set_hub_categories(data[3])
            hub.set_is_semantic(data[4])

            hub_list.append(hub)
        return hub_list

    def update_hub_list(self):
        """
        (2022.01.07 통합용 추가)
        통합 검색이 가능한 데이터 허브들을 업데이트함 (make_search_hub_list 전에 수행됨)
        :return: self.hub_list가 업데이트 됨.
        """

        query = "select * from " + config['federated_member_hub_table']
        self.cursor.execute(query)
        result = self.cursor.fetchall()

        update_hub_list = []

        for data in result:
            hub = DataHubInfo()
            hub.set_hub_url(data[2])
            hub.set_hub_name(data[1])
            hub.set_hub_categories(data[3])
            hub.set_is_semantic(data[4])

            update_hub_list.append(hub)
        self.hub_list = update_hub_list

    def federated_query_intent_analysis(self, intents, threshold=0.6):
        """
        통합된 사용자 의도 재분석을 수행하고, threshold를 넘기는 결과를 반환함

        :param intents: 각 허브에서 받아온 사용자 의도 분석 결과
        :param threshold: 통합 모델 threshold
        :return: 사용자 의도 분석 결과.
        """

        result = intents.copy()
        for keyword in intents:
            try:
                intents = self.integrate_model.wv.most_similar(keyword)
                for name, score in intents:
                    if score >= threshold:
                        if name in result:
                            result[name] = max(result[name], score)
                        else:
                            result[name] = score
            except:
                print("추가 확장 결과 없음")
                pass

        return result

    def make_search_hub_list(self, categories="", sites=""):
        """
        사용자 입력(categories, sites)에 따라 통합 검색을 수행할 데이터 허브 리스트를 만듦

        :param categories: (list(string))데이터 허브의 카테고리("의료", "교통" 등)
        :param sites: (list(string))데이터 허브의 주소
        :return: (list(DataHubInfo))사용자 입력 조건에 맞는, 통합 검색을 수행할 데이터 허브 리스트
        """

        self.update_hub_list()  # 통합 검색을 수행하기 위한 포털을 불러오기 이전에 업데이트를 수행함.

        if categories is None:
            categories = ""
        if sites is None:
            sites = ""

        if categories == "" and sites == "":
            print("지정된 사이트 없음")
            return self.hub_list

        search_hub_list = []

        if categories != "" and sites == "":
            print("카테고리 있음")
            categories = categories.split()
            for hub in self.hub_list:
                for category in categories:
                    if category in hub.get_hub_categories():  # 도메인 중에 있는 경우
                        search_hub_list.append(hub)

        elif sites != "" and categories == "":
            print("사이트 있음")
            sites = sites.split()
            for hub in self.hub_list:
                if hub.get_hub_url() in sites:  # sites 중에 일치하는 도메인이 있는 경우
                    search_hub_list.append(hub)
        else:
            print("둘다 있음")
            categories = categories.split()
            sites = sites.split()
            for hub in self.hub_list:
                for category in categories:
                    if category in hub.get_hub_categories():  # 도메인 중에 있는 경우
                        search_hub_list.append(hub)
                    elif hub.get_hub_url() in sites:  # sites 중에 일치하는 도메인이 있는 경우
                        search_hub_list.append(hub)

        return search_hub_list

    def do_query_intent_analysis(self, search_hub_list, query, algorithm_id="", algorithm_parameter={}):
        """
        통합 검색을 수행할 데이터 허브들에게 사용자 질의 의도 분석 요청을 보내는 함수

        :param search_hub_list: (list(DataHubInfo))통합 검색을 수행할 데이터 허브 목록
        :param query: (string)사용자 질의
        :param algorithm_id: (string)의도 분석 알고리즘 아이디
        :param algorithm_parameter: 의도 분석 알고리즘 파라미터
        :return: 각 포털 별 사용자 의도 분석 결과
        """

        # 사용자 의도 통합 결과 저장 리스트
        total_intent = []

        # POST 요청을 위해서 algorithm_parameter -> json 변경해서 넘겨야 함.
        algorithm_parameter = json.dumps(algorithm_parameter, ensure_ascii=False)
        #print(query, type(query))
        # 동시 request 처리를 위한 튜플을 저장하는 변수
        list_of_requests = []

        # algorithm_id가 있는 경우
        if algorithm_id == '1':
            # 요청 보낼 url, body 튜플 생성
            # 만약 검색에 포함된 허브가 의도 분석이 불가능 할 경우
            for hubinfo in search_hub_list:
                # 요청
                if hubinfo.get_is_semantic() is True:
                    url = str(hubinfo.get_hub_url())
                    list_of_requests.append(
                        ('http://' + url + '/discovery/semantic/intent-based/intent/get?searchValue=' + query \
                        + '&algorithmId=' + algorithm_id, algorithm_parameter))
                else:
                    continue
        # algorithm_id가 없는 경우
        else:
            for hubinfo in search_hub_list:
                # 요청
                if hubinfo.get_is_semantic() is True:
                    url = hubinfo.get_hub_url()
                    list_of_requests.append(
                        ('http://' + url + '/discovery/semantic/intent-based/intent/get?searchValue=' + query \
                            , algorithm_parameter))
                else:
                    continue

        # 위에서 분기에 따라 매핑한 requests list에 따라 post 실행
        with ThreadPoolExecutor(max_workers=10) as pool:
            response_list = list(pool.map(post_url, list_of_requests))

        # 데이터를 파싱해서 total_intent에 넣음.
        for response in response_list:
            if isinstance(response, requests.models.Response) and response.status_code is 200:
                pass
            else:
                print("IS NOT RESPONSE")
                continue
            response = response.json()
            temp_dic = {}
            for data in response['result']:
                temp_value_list = []
                for intent in data['intents']:
                    # {'intent': 'string', 'score': integer}
                    # => ['string', integer]
                    temp = list(intent.values())
                    temp_value_list.append(temp)

                key = data['keyword']
                temp_dic[key] = temp_value_list

            total_intent.append(temp_dic)
        return total_intent

    def integrate_intent(self, intents, threshold=0.55, total_similarity=0.6):
        """
        각 시맨틱 데이터 허브의 의도 분석 결과를 통합하고 재분석 결과를 반환하는 함수

        :param intents: 의도 분석 결과
        :param threshold: 각 허브 별 의도 분석 결과 통합 시 threshold
        :param total_similarity: 통합 모델 재분석 결과에 대한 threshold
        :return: 각 의도 분석 통합 결과 + 재분석 결과
        """
        key_list = []
        integrate_results = {}

        # intents에서 key 추출
        for intent in intents:
            keys = intent.keys()
            for key in keys:
                if key not in key_list:
                    key_list.append(key)

        # 사용자 질의 값을 integrate_result에 1.0의 score로 추가함.
        for key in key_list:
            if key not in integrate_results:
                integrate_results[key] = {key: 1.0}

        # 각 허브의 사용자 의도 분석 결과를 추출함.
        for intent in intents:
            for key in key_list:
                data = intent[key]
                for item in data:
                    name, score = item
                    score = float(score)
                    # 의도 분석 결과 점수가 threshold 미만이면 통합에 포함하지 않음
                    if score < threshold:
                        continue
                    # 중복된 결과가 있는 경우 max로 통합
                    if name in integrate_results[key]:
                        integrate_results[key][name] = max(integrate_results[key][name], score)
                    else:
                        # 중복 결과가 없으면 추가
                        integrate_results[key][name] = score

        # 병합된 의도 재분석 수행
        for keyword in integrate_results:
            integrate_results[keyword] = self.federated_query_intent_analysis(integrate_results[keyword],
                                                                                total_similarity)

        # 통합 사용자 의도 분석 결과 반환
        return integrate_results

    def do_search(self, search_hub_list, query):
        """
        통합 검색 수행 허브 목록에 있는 포털에 의도 분석 결과로 검색 요청을 보내고, 중복을 제거한 검색 결과를 반환함.

        :param search_hub_list: 통합 검색을 수행할 데이터 허브 목록
        :param query: 사용자 질의
        :return:  각 포털 별 검색 통합 결과
        """

        search_results = []

        # 내부 api에 맞추기 위해 query를 한번 감싼다.
        query = {"query": query}

        # {"query": [[서울특별시, 교통], [광진구, 교통]...]}
        query_json = json.dumps(query, ensure_ascii=False)

        # 동시 request 처리를 위한 튜플 리스트 저장하는 변수
        list_of_requests = []

        # 요청 보낼 url, body 튜플 생성
        for url in search_hub_list:
            url = url.get_hub_url()
            list_of_requests.append(('http://' + url + '/discovery/keyword/search', query_json.encode('utf-8')))

        with ThreadPoolExecutor(max_workers=10) as pool:
            response_list = list(pool.map(post_url, list_of_requests))

        for response in response_list:
            # 요청이 올바른 반환을 줄 경으 pass.
            if isinstance(response, requests.models.Response) and response.status_code is 200:
                pass
            else:
                continue
            result = response.json()
            # 검색 결과가 있는 경우 결과에 포함.
            if len(result['results']) > 0:
                search_results.extend(result['results'])
            else:
                print(url, "허브의 결과 스킵")

        print("403::::", search_results)
        # 검색 결과 중복 제거
        integrate_result = self.integrate_search_result(search_results)

        return integrate_result

    def integrate_search_result(self, hubs_datasets):
        """
        각 데이터 허브로부터 전달 받은 전체 검색 결과의 중복을 제거함.

        :param hubs_datasets: 중복을 제거할 전체 검색 결과
        :return: 중복을 제거한 hub_datasets
        """
        deduplicate_list = []
        # accecssURL을 기준으로 중복을 제거함. 다른 값을 넣으면 해당 값으로 중복 판별 수행
        if len(hubs_datasets) > 0:
            deduplicate_list = list({data['uid']: data for data in hubs_datasets}.values())

        return deduplicate_list

    def federated_search_ctrl(self, datahubs, query, body={}):
        """
        body 값에 따라 알고리즘 파라미터를 조정해 통합 검색을 수행함

        :param datahubs:(list(DataHubInfo))통합 검색을 할 datahubs 목록
        :param query: (string)사용자 질의
        :param body: algorithm_parameter
        :return: 통합 검색 수행 결과(중복 제거 이전)
        """
        # body의 파라미터 파싱.
        datahub_parameter = {"parameter": [{"similarity": "0.5"}]} if 'parameter' not in body \
            else {"parameter": body['parameter']}
        if 'totalParameter' in body:
            threshold = 0.55 if 'threshold' not in body['totalParameter'][0] else float(
                body['totalParameter'][0]['threshold'])
            similarity = 0.6 if 'similarity' not in body['totalParameter'][0] else float(
                body['totalParameter'][0]['similarity'])
        else:
            threshold = 0.55
            similarity = 0.6

        # 데이터 허브에서 사용할 파라미터 적용
        intents = self.do_query_intent_analysis(datahubs, query, '1', datahub_parameter)
        # 재분석 시 사용할 파라미터 적용
        integrated_intents = self.integrate_intent(intents, threshold, similarity)
        intent_query = self.make_query(integrated_intents)
        result_datasets = self.do_search(datahubs, intent_query)
        return result_datasets

    def get_federated_intent(self, datahubs, query, body={}):
        """
        사용자 질의에 대한 통합 사용자 의도 분석 결과를 반환하는 함수

        :param datahubs:(list(DataHubInfo))통합 검색을 할 datahubs 목록
        :param query: (string)사용자 질의
        :param body: algorithm_parameter
        :return: 통합 사용자 의도 분석 결과
        """

        # body의 파라미터 파싱.
        datahub_parameter = {"parameter": [{"similarity": "0.5"}]} if 'parameter' not in body \
            else {"parameter": body['parameter']}
        if 'totalParameter' in body:
            threshold = 0.55 if 'threshold' not in body['totalParameter'][0] else float(
                body['totalParameter'][0]['threshold'])
            similarity = 0.6 if 'similarity' not in body['totalParameter'][0] else float(
                body['totalParameter'][0]['similarity'])
        else:
            threshold = 0.55
            similarity = 0.6

        # 데이터 허브에서 사용할 파라미터 적용
        intents = self.do_query_intent_analysis(datahubs, query, '1', datahub_parameter)
        # 재분석 시 사용할 파라미터 적용
        integrated_intents = self.integrate_intent(intents, threshold, similarity)

        result_list = []

        for key in integrated_intents.keys():
            result_dic = {}
            result_dic['keyword'] = key
            intents_list = []
            keys = list(integrated_intents[key].keys())
            values = list(integrated_intents[key].values())
            for i in range(0, len(integrated_intents[key])):
                temp_dic = {}
                temp_dic['intent'] = keys[i]
                temp_dic['score'] = values[i]
                intents_list.append(temp_dic)
            result_dic['intents'] = intents_list

            result_list.append(result_dic)

        return result_list

    def re_search(self, search_hub_list, intent_list):
        """
        사용자가 선택한 키워드로 elastic search form의 질의를 생성해 통합 재검색을 수행하는 함수

        :param intent_list: 재검색을 수행할 키워드
        :return: 검색 결과 반환 (list of dicts)
        """

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

        

        query_list = list(product(*facet_list))

        '''
        query_dict = {}
        query_dict["query"] = list(product(*facet_list))

        
        # str list sub element join with and, str list element join with or
        modified_query_list = []
        for elem in query_dict["query"]:
            sub_body = " AND ".join(elem)
            modified_query_list.append("(" + sub_body + ")")

        body = " OR ".join(modified_query_list)
        
        print("body",body)
        
        # es body
        result_body = {
            "query": {
                "nested": {
                    "path": "basicMetadata.value",
                    "query": {
                        "query_string": {
                            "query": body,
                            "fields": ["*"],
                            "default_operator": "AND"
                        }
                    }}}
        }
        result_datasets = self.do_search(search_hub_list, result_body)
        '''

        result_datasets = self.do_search(search_hub_list, query_list)

        return result_datasets

    def get_federated_search_aggs(self, datasets, keys=["publisher","modified"]):
        """
        통합 검색 결과에 대한 통계를 반환하는 함수로, 입력 받은 데이터셋 리스트에 대한 통계를 아래와 같이 반환함
        {
          'property1':[
            {
              'key': 'key_name1',
              'doc_count': 15
            },
            {
              'key': 'key_name2',
              'doc_count': 12
            }
          ],
          'property2':[
            {
              'key': 'key_name1',
              'doc_count': 15
            },
            {
              'key': 'key_name2',
              'doc_count': 12
            }
          ]
        }

        :param datasets: 통계치를 생성할 데이터셋 리스트
        :param keys: 통계를 수집할 property key 이름
        :return aggs: 반환할 통계 수치를 담은 dict
        """

        from collections import Counter

        aggs = {}

        for key in keys:
            agg = dict(Counter(d[key] for d in datasets))
            aggs[key] = agg

        parsing_data = {}
        for key, value in aggs.items():
            parsing_data[key] = [{'key': k, 'doc_count': v} for k, v in value.items()]

        return parsing_data