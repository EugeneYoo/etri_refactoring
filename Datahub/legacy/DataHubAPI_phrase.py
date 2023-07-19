import json
import math

import os

from flask import Flask
from flask import request
from flask import make_response
from flask_cors import CORS

from Datahub.src.classes.datahub_phrase import DataHub
#from ..classes.pagination_phrase import Pagination
from Datahub.src.classes.pagination_multi import Pagination
from Datahub.src.classes.facet import Facet

from Datahub.src.classes.config_phrase import args

import ranking as rankService
from Datahub.src.ranker import Ranker

from operator import itemgetter

ranker = Ranker.get_instance()

import requests

from werkzeug.middleware.profiler import ProfilerMiddleware

from Datahub.src.classes.error_handler import error_handle

#from waitress import serve
# desc: hub 객체를 초기화 하기 위한 config dictionary
# input
# model: 데이터 허브의 의도 분석 모델(word2vec bin)파일 - 추가 모델은 model 폴더 참조
# elasticsearch: hub dataset이 저장된 elasticsearch 서버 주소
# hub_index_name: hub의 dataset이 저장된 es index name

env = 'development' if not 'APP_ENV' in os.environ else os.environ['APP_ENV']
config = args[env]

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['UPLOAD_FOLDER'] = './upload_data'
app.wsgi_app = ProfilerMiddleware(app.wsgi_app)

CORS(app)
error_handle(app)

print(config['hub_index_name'])
# api를 사용할 객체 선언
hub = DataHub(hub_index_name=config['hub_index_name'], elasticsearch=config['elasticsearch'], use_ssl=config['use_ssl'],
                verify_certs=config['verify_certs'], model=config['query_model'])

# 시맨틱 검색 페이지를 pagination 객체 선언
pagination = Pagination()

# 모델 아이디 관리를 위한 global
w2vmodel_id = 5
remodel_id = 1


# 통합 검색 수행 시 호출되는 keyword search API
# 통합 검색을 위한 내부 API
@app.route("/discovery/keyword/search", methods=['POST'])
def search_by_keyword():
    body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))

    query = body['query']

    # str list sub element join with and, str list element join with or
    modified_query_list = []
    for elem in query:
        sub_body = " AND ".join(elem)
        modified_query_list.append("(" + sub_body + ")")

    body = " OR ".join(modified_query_list)

    search_result = hub.search(body)
    result = {'results': search_result}
    result = json.dumps(result, ensure_ascii=False).encode('utf-8')
    res = make_response(result)

    return res

# 단일 포털 검색(semantic search) 수행 API
@app.route("/discovery/semantic/intent-based/search", methods=['POST'])
def main_search():
    """
    pr= cProfile.Profile
    pr.enable()
    """
    query = request.args.get('searchValue')
    dataset_count = request.args.get('datasetCount', default=10, type=int)
    algorithm_id = request.args.get('algorithmId')

    dataset_count = int(dataset_count)

    # 추가된 부분
    # 값이 오면 사용하고, 없는 경우 1을 defalut로 사용함.
    page_num = request.args.get('pageNum', default='1', type=str)
    page_num = int(page_num)

    start = dataset_count * (page_num - 1) + 1
    end = start + dataset_count - 1


    # log_data = json.dumps({"log_data": {"query": query, "algorithm":{"algorithm_id": algorithm_id}}})
    # requests.post('http://163.180.116.87:5000/query_log_api', log_data)

    # 모델 변경 여부 체크
    if (hub.intent_analyzer.modi_check_and_reload_model()):
        # 변경되면 flush
        pagination.flush_old_data()

    # 단순 키워드 검색
    if algorithm_id == '0':
        cache_dic = {}
        cache_dic['query'] = query
        cache_dic['algorithm_id'] = '0'
        cache_key = str(cache_dic)

        # 캐시가 있는 경우
        if pagination.is_cached(cache_key):
            result, total_len = pagination.get_data(cache_key, start, end)
        else:
            # 캐시가 없는 경우
            # str list sub element join with and, str list element join with or
            query = query.split()
            print(query)
            
            sub_body = " AND ".join(query)
            sub_body = "(" + sub_body + ")"

            search_result = hub.search(sub_body)
            pagination.insert_data(cache_key, search_result)
            result, total_len = pagination.get_data(cache_key, start, end)

    # threshold 알고리즘 파라미터를 적용한 검색
    elif algorithm_id == '1':
        body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))
        # 캐시가 있는지 확인할 key_dictionary 생성
        cache_dic = {}
        cache_dic['query'] = query
        cache_dic['body'] = body
        cache_key = str(cache_dic)
        # 캐시가 있는 경우
        if pagination.is_cached(cache_key):
            result, total_len = pagination.get_data(cache_key, start, end)

            resultids = []
            for data in result:
                resultids.append(data['uid'])
            log_data = {
                "logzone": "DataSearch",
                "loggerName": "ListLogger",
                "message": {
                    "datasetList": resultids
                }
            }
            jlog_data = json.dumps(log_data, ensure_ascii=False)
            if (resultids != 0):
                requests.post('http://163.180.116.87:5000/log/list/collect', \
                                data=jlog_data)
        # 캐시가 없는 경우
        else:
            threshold = float(body['parameter'][0]['similarity'])
            dataset = hub.semantic_search(query, algorithm_id=algorithm_id, algorithm_parameter=threshold)
            pagination.insert_data(cache_key, dataset)
            result, total_len = pagination.get_data(cache_key, start, end)

            resultids = []
            for data in result:
                resultids.append(data['uid'])
            log_data = {
                "logzone": "DataSearch",
                "loggerName": "ListLogger",
                "message": {
                    "datasetList": resultids
                }
            }
            jlog_data = json.dumps(log_data, ensure_ascii=False)
            if (resultids != 0):
                requests.post('http://163.180.116.87:5000/log/list/collect', \
                                data=jlog_data)

    elif algorithm_id == "2":
        # rerank 임시 적용
        body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))
        # 캐시가 있는지 확인할 key_dictionary 생성
        cache_dic = {}
        cache_dic['query'] = query
        cache_dic['body'] = body
        cache_key = str(cache_dic)
        # 캐시가 있는 경우
        if pagination.is_cached(cache_key):
            result_all = pagination.get_all_data(cache_key)
            total_len = len(result_all)

            rerank_result = rankService.getRank(query, result_all)

            ranker.setResultDataset(rerank_result)

            rerank_result = sorted(rerank_result, key=itemgetter('finalScore', 'uid'), reverse=True)

            result = []
            if len(rerank_result) < end:
                end = len(rerank_result)
            for i in range(start - 1, end):
                result.append(rerank_result[i])

            # 결과 출력을 위해 d2v 제거
            for i in range(0, len(result)):
                result[i].pop('documentVector')

            resultids = []
            for data in result:
                resultids.append(data['uid'])
            log_data = {
                "logzone": "DataSearch",
                "loggerName": "ListLogger",
                "message": {
                    "datasetList": resultids
                }
            }
            jlog_data = json.dumps(log_data, ensure_ascii=False)
            if (resultids != 0):
                requests.post('http://163.180.116.87:5000/log/list/collect', \
                                data=jlog_data)

        # 캐시가 없는 경우
        else:
            cache_dic = {}
            cache_dic['query'] = query
            # default 값이 0.5이므로, body 로 0.5를 넣어줌.
            body_value = {'parameter': [{'similarity': 0.5}]}
            cache_dic['body'] = body_value
            cache_key = str(cache_dic)
            threshold = float(body_value['parameter'][0]['similarity'])
            dataset = hub.semantic_search(query, algorithm_id=algorithm_id, algorithm_parameter=threshold)
            pagination.insert_data(cache_key, dataset)
            result_all = pagination.get_all_data(cache_key)
            total_len = len(result_all)
            resultids = []

            # 임시 재정렬 코드
            #slice_data = result_all[:30]
            slice_data = result_all
            rerank_result = rankService.getRank(query, slice_data)

            ranker.setResultDataset(rerank_result)

            rerank_result = sorted(rerank_result, key=itemgetter('finalScore', 'uid'), reverse=True)

            result = []
            if len(rerank_result) < end:
                end = len(rerank_result)
            for i in range(start - 1, end):
                result.append(rerank_result[i])

            # 결과 출력을 위해 d2v 제거
            for i in range(0, len(result)):
                result[i].pop('documentVector')

            for data in result:
                resultids.append(data['uid'])
            log_data = {
                "logzone": "DataSearch",
                "loggerName": "ListLogger",
                "message": {
                    "datasetList": resultids
                }
            }
            jlog_data = json.dumps(log_data, ensure_ascii=False)
            if (resultids != 0):
                requests.post('http://163.180.116.87:5000/log/list/collect', \
                                data=jlog_data)

    else:
        cache_dic = {}
        cache_dic['query'] = query
        # default 값이 0.5이므로, body 로 0.5를 넣어줌.
        body_value = {'parameter': [{'similarity': 0.5}]}
        cache_dic['body'] = body_value
        cache_key = str(cache_dic)
        # 캐시가 있는 경우
        if pagination.is_cached(cache_key):
            print(cache_key)
            result, total_len = pagination.get_data(cache_key, start, end)

            resultids = []
            for data in result:
                resultids.append(data['uid'])
            log_data = {
                "logzone": "DataSearch",
                "loggerName": "ListLogger",
                "message": {
                    "datasetList": resultids
                }
            }
            jlog_data = json.dumps(log_data, ensure_ascii=False)
            if (resultids != 0):
                requests.post('http://163.180.116.87:5000/log/list/collect', \
                              data=jlog_data)

        # 캐시가 없는 경우
        else:
            print(cache_key)
            dataset = hub.semantic_search(query)
            pagination.insert_data(cache_key, dataset)
            result, total_len = pagination.get_data(cache_key, start, end)

            resultids = []
            for data in result:
                resultids.append(data['uid'])
            log_data = {
                "logzone": "DataSearch",
                "loggerName": "ListLogger",
                "message": {
                    "datasetList": resultids
                }
            }
            jlog_data = json.dumps(log_data, ensure_ascii=False)
            if (resultids != 0):
                requests.post('http://163.180.116.87:5000/log/list/collect', \
                              data=jlog_data)
    """

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    """
    result_map = {
        "totalPage": math.ceil(total_len / dataset_count),
        "currentPage": page_num,
        "results": result}
    return json.dumps(result_map, ensure_ascii=False)


# query를 통해 사용자 의도 분석을 수행하고 점수를 포함한 결과를 반환,
# 통합 검색을 위한 내부 API, API 문서에 기술 X
@app.route("/discovery/semantic/intent-based/intent/get", methods=['POST'])
def analyze_intent_with_score():
    query = request.args.get('searchValue')
    algorithm_id = request.args.get('algorithmId')

    print("query: ", query, "aid:", algorithm_id)

    # parameter 사용, 원하는 threshold 적용
    if algorithm_id == '1':
        body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))

        threshold = float(body['parameter'][0]['similarity'])
        intents = hub.analyze_intent_with_score(query, threshold)
    # parameter 사용 X, 디폴트 threshold 적용(0.5)
    else:
        intents = hub.analyze_intent_with_score(query)

    result = {'result': intents}
    result_json = json.dumps(result, ensure_ascii=False).encode('utf-8')
    res = make_response(result_json)

    return res


# 의도 분석 결과 중 사용자가 선택한 intent keyword로 재검색을 수행하는 API
@app.route("/discovery/semantic/intent-based/intent/search", methods=['POST'])
def research_by_intent():
    body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))

    dataset_count = request.args.get('datasetCount', default=10, type=int)
    dataset_count = int(dataset_count)

    # 추가된 부분
    # 값이 오면 사용하고, 없는 경우 1을 defalut로 사용함.
    page_num = request.args.get('pageNum', default='1', type=str)
    page_num = int(page_num)

    start = dataset_count * (page_num - 1) + 1
    end = start + dataset_count - 1

    query = body['query']
    if len(query) == 0:
        return 'No keywords in.'

    cache_dic = {}
    cache_dic['query'] = query
    cache_key = str(cache_dic)

    # 캐시가 있는 경우
    if pagination.is_cached(cache_key):
        result, total_len = pagination.get_data(cache_key, start, end)
    else:
        result_datasets = hub.re_search(query)
        pagination.insert_data(cache_key, result_datasets)
        result, total_len = pagination.get_data(cache_key, start, end)

    result_map = {"totalPage": math.ceil(total_len / dataset_count),
                  "currentPage": page_num,
                  "results": result}

    result = json.dumps(result_map, ensure_ascii=False).encode('utf-8')
    res = make_response(result)

    return res


# desc: 통합 검색 결과에서 facet을 추출함. 현재 알고리즘이 없어서 기본 설정으로만 동작함.
@app.route("/discovery/filter/facet/get", methods=['POST'])
def get_facets():
    query = request.args.get('searchValue')
    algorithm_id = request.args.get('algorithmId')

    result_map = {}

    # 현재 알고리즘이 없으므로 분기만 설정함.
    if algorithm_id is None:
        cache_dic = {}
        cache_dic['query'] = query
        # default 값이 0.5이므로, body 로 0.5를 넣어줌.
        body_value = {'parameter': [{'similarity': 0.5}]}
        cache_dic['body'] = body_value
        cache_key = str(cache_dic)

        if pagination.is_cached(cache_key):
            original_datasets = pagination.get_all_data(cache_key)

            # 불러온 경우 해당 데이터로 facet 객체 생성
            facet_obj = Facet(original_datasets)

        else:
            return json.dumps("잘못된 키가 들어왔습니다.", ensure_ascii=False)
    elif algorithm_id is '0':

        cache_dic = {}
        cache_dic['query'] = query
        cache_dic['algorithm_id'] = '0'
        cache_key = str(cache_dic)

        cache_key = str(cache_dic)

        if pagination.is_cached(cache_key):
            original_datasets = pagination.get_all_data(cache_key)

            # 불러온 경우 해당 데이터로 facet 객체 생성
            facet_obj = Facet(original_datasets)
        else:
            return json.dumps("잘못된 키가 들어왔습니다.", ensure_ascii=False)

    else:
        # 현재 알고리즘은 없음.
        body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))

        cache_dic = {}
        cache_dic['query'] = query
        cache_dic['body'] = body
        print(cache_dic)
        cache_key = str(cache_dic)

        if pagination.is_cached(cache_key):
            original_datasets = pagination.get_all_data(cache_key)

            # 불러온 경우 해당 데이터로 facet 객체 생성
            facet_obj = Facet(original_datasets)
        else:
            return json.dumps("잘못된 키가 들어왔습니다.", ensure_ascii=False)

    result_map["facets"] = facet_obj.facet_wrapper_for_response()
    res = json.dumps(result_map, ensure_ascii=False)
    return res


# desc: facet filter를 적용한 검색 결과를 출력함
@app.route("/discovery/filter/facet/apply", methods=['POST'])
def apply_facet_filter():
    query = request.args.get('searchValue')
    algorithm_id = request.args.get('algorithmId')
    dataset_count = request.args.get('datasetCount', default=10, type=int)
    dataset_count = int(dataset_count)

    # 추가된 부분
    # 값이 오면 사용하고, 없는 경우 1을 defalut로 사용함.
    page_num = request.args.get('pageNum', default='1', type=str)
    page_num = int(page_num)

    start = dataset_count * (page_num - 1) + 1
    end = start + dataset_count - 1

    body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))
    facets = body["facets"]

    # 현재 알고리즘이 없으므로 분기만 설정함.
    if algorithm_id is None:
        cache_dic = {}
        cache_dic['query'] = query
        # default 값이 0.5이므로, body 로 0.5를 넣어줌.
        body_value = {'parameter': [{'similarity': 0.5}]}
        cache_dic['body'] = body_value
        cache_key = str(cache_dic)

        print(cache_key)

        if pagination.is_cached(cache_key):
            original_datasets = pagination.get_all_data(cache_key)

            # 불러온 경우 해당 데이터로 facet 객체 생성
            facet_obj = Facet(original_datasets)

        else:
            return json.dumps("잘못된 키가 들어왔습니다.", ensure_ascii=False)

    else:
        # 현재 알고리즘은 없음.
        body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))

        cache_dic = {}
        cache_dic['query'] = query
        temp = {"parameter": body['parameter']}
        cache_dic['body'] = temp

        print(cache_dic)
        cache_key = str(cache_dic)

        if pagination.is_cached(cache_key):
            original_datasets = pagination.get_all_data(cache_key)

            # 불러온 경우 해당 데이터로 facet 객체 생성
            facet_obj = Facet(original_datasets)
        else:
            return json.dumps("잘못된 키가 들어왔습니다.", ensure_ascii=False)

    # 사용자가 선택한 facet을 적용함.
    facet_obj.apply_facet_filter(facets)
    # 변경된 filtered_datasets를 불러옴.
    datasets = facet_obj.get_filtered_datasets()

    result = []
    if len(datasets) < end:
        end = len(datasets)
    for i in range(start - 1, end):
        result.append(datasets[i])

    print("데이터셋 크기:", len(datasets))
    print("데이터셋 카운트 크기:", dataset_count)

    # 리턴할 결과 맵
    result_map = {"totalPage": math.ceil(len(datasets) / dataset_count),
                  "currentPage": page_num,
                  "filteredDatasets": result}

    return json.dumps(result_map, ensure_ascii=False)


@app.route('/discovery/semantic/intent-based/search/aggs', methods=['POST'])
def get_query_aggs():
    query = request.args.get('searchValue')
    algorithm_id = request.args.get('algorithmId')

    # threshold 알고리즘 파라미터로 적용한 검색
    if algorithm_id == '1':
        body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))
        # 캐시가 있는지 확인할 key_dictionary 생성

        threshold = float(body['parameter'][0]['similarity'])
        aggs = hub.get_search_aggs(query, algorithm_id=algorithm_id, algorithm_parameter=threshold)
    else:
        aggs = hub.get_search_aggs(query)

    return json.dumps(aggs, ensure_ascii=False).encode('utf8')


@app.route('/discovery/semantic/intent-based/intent/search/aggs', methods=['POST'])
def get_re_searhc_aggs():
    body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))

    query = body['query']
    if len(query) == 0:
        return 'No keywords in.'

    aggs = hub.get_re_search_aggs(query)

    return json.dumps(aggs, ensure_ascii=False).encode('utf8')

@app.route('/model/original')
def change_model_ori():
    from gensim.models import KeyedVectors
    hub.intent_analyzer.pretrained_model = KeyedVectors.load_word2vec_format(r'C:\Users\hi\Desktop\khu_2\khu\Datahub\model\phrase\ko_combine.bin', binary=True,
                                                                            unicode_errors='ignore')
    hub.intent_analyzer.model_path = r'C:\Users\hi\Desktop\khu_2\khu\Datahub\model\phrase\ko_combine.bin'

    return "오리지널 모델"


@app.route('/model/query')
def change_model_query():
    from gensim.models import KeyedVectors
    hub.intent_analyzer.pretrained_model = KeyedVectors.load_word2vec_format('./query_txt.bin', binary=True, unicode_errors='ignore')
    hub.intent_analyzer.model_path = './query_txt.bin'

    return "쿼리 학습 모델"

@app.route('/model/query/<id>')
def change_model_query_with_id(id):
    from gensim.models import KeyedVectors

    path = f'./data/w2v_{id}.bin'
    hub.intent_analyzer.pretrained_model = KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors='ignore')
    hub.intent_analyzer.model_path = path
    return "학습 모델"+ path

@app.route('/')
def main_page():
    print("hit main page")
    return {"success": True, "msg": "api 주소를 입력해주세요"}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8500, debug=False)
    #serve(app, host='0.0.0.0', port=8500, url_prefix='/search')
