import math
import json

from flask import Flask
from flask import request
from flask import make_response
from flask_cors import CORS

from gensim.models import Word2Vec, KeyedVectors

from classes.federateddiscoveryengine import FederatedDiscoveryEngine
from classes.pagination import Pagination
from classes.facet import Facet
from classes.error_handler import error_handle

# total_model: 추가학습 시 사용할 통합 모델 주소
config = {
    'total_model': './model/ko_latest_통합모델.bin'
}

app = Flask(__name__)

CORS(app)
error_handle(app)

app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

pagination = Pagination()
federated_engine = FederatedDiscoveryEngine()

# desc: 통합 검색을 위한 categories 목록 불러오는 함수, API 문서 참조
@app.route("/discovery/federated/categories")
def get_categories():

    total_categories = []
    federated_engine.update_hub_list()
    # 전체 데이터 허브 리스트를 받음.
    for hub in federated_engine.hub_list:
        # 한 데이터 포털의 카테고리 받음
        hub_categories = hub.get_hub_categories()
        for category in hub_categories:
            if category not in total_categories:
                total_categories.append(category)
            else:
                pass

    result = json.dumps(total_categories, ensure_ascii=False).encode('utf-8')

    res = make_response(result)

    return res


# desc: 통합 검색을 위한 sites 목록 불러오는 함수, API 문서 참조
@app.route("/discovery/federated/sites")
def load_sites():
    total_urls = []
    federated_engine.update_hub_list()
    # 전체 데이터 허브 리스트를 받음.
    for hub in federated_engine.hub_list:
        # 한 데이터 포털의 url 받음
        hub_url = hub.get_hub_url()
        if hub_url not in total_urls:
            total_urls.append(hub_url)
        else:
            pass

    result = json.dumps(total_urls, ensure_ascii=False).encode('utf-8')

    res = make_response(result)

    return res


# desc: 통합 검색을 수행하고 검색 결과를 반환함
@app.route("/discovery/federated/intent-based/search", methods=['POST'])
def federated_search():
    query = request.args.get('searchValue')
    categories = request.args.get('categories')
    sites = request.args.get('sites')
    dataset_count = request.args.get('datasetCount', default=10, type=int)
    algorithm_id = request.args.get('algorithmId')
    dataset_count = int(dataset_count)

    # 추가된 부분
    # 값이 오면 사용하고, 없는 경우 1을 defalut로 사용함.
    page_num = request.args.get('pageNum', default='1', type=str)
    page_num = int(page_num)

    start = dataset_count * (page_num - 1) + 1
    end = start + dataset_count - 1

    datahubs = federated_engine.make_search_hub_list(categories, sites)

    print(datahubs)

    # algorithm_parameter는 다음과 같으며 3가지를 조정가능, 사용하지 않을 파라미터는 쓰지 않아도 동작함.
    # parameter similarity: 각각의 시맨틱 허브의 의도 분석 결과 유사도 점수 기준
    # totalParameter threshold: 각 허브 의도 통합 시 분석 결과에 포함시키기 위한 기준
    # totalParameter similarity: 재분석 결과의 유사도 점수 기준
    # body = {
    # "parameter":[{"similarity":"0.5"}],
    # "totalParameter":[{"threshold":"0.6","similarity":"0.6"}]
    # }
    if algorithm_id == '1':
        body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))

        # 캐시가 있는지 확인할 key_dictionary 생성
        cache_dic = {}
        cache_dic['query'] = query
        cache_dic['body'] = body
        cache_dic['portal'] = datahubs
        cache_key = str(cache_dic)

        # 캐시가 있는 경우
        if pagination.is_cached(cache_key):
            result, total_len = pagination.get_data(cache_key, start, end)
        else:
            # 통합 검색 수행
            result_datasets = federated_engine.federated_search_ctrl(datahubs, query, body)
            pagination.insert_data(cache_key, result_datasets)
            result, total_len = pagination.get_data(cache_key, start, end)
    else:
        # 캐시가 있는지 확인할 key_dictionary 생성
        cache_dic = {}
        cache_dic['query'] = query
        # default 값을 가진 body 생성
        body_value = {'parameter': [{'similarity': '0.5'}],
                        'totalParameter': [{'threshold': '0.55', 'similarity': '0.6'}]}
        cache_dic['body'] = body_value
        cache_dic['portal'] = datahubs
        cache_key = str(cache_dic)
        # 캐시가 있는 경우
        if pagination.is_cached(cache_key):
            print(cache_key)
            result, total_len = pagination.get_data(cache_key, start, end)
        # 캐시가 없는 경우
        else:
            result_datasets = federated_engine.federated_search_ctrl(datahubs, query)
            pagination.insert_data(cache_key, result_datasets)
            result, total_len = pagination.get_data(cache_key, start, end)

    result_map = {"totalPage": math.ceil(total_len / dataset_count),
                    "currentPage": page_num,
                    "results": result}

    return json.dumps(result_map, ensure_ascii=False)


# desc: 통합 사용자 의도 분석 결과를 보여줌
@app.route("/discovery/federated/intent-based/intent/get", methods=['POST'])
def get_federated_intent():
    query = request.args.get('searchValue')
    categories = request.args.get('categories')
    sites = request.args.get('sites')
    algorithm_id = request.args.get('algorithmId')

    datahubs = federated_engine.make_search_hub_list(categories, sites)

    # algorithm_parameter는 다음과 같으며 3가지를 조정가능, 사용하지 않을 파라미터는 쓰지 않아도 동작함.
    # parameter similarity: 각각의 시맨틱 허브의 의도 분석 결과 유사도 점수 기준
    # totalParameter threshold: 각 허브 의도 통합 시 분석 결과에 포함시키기 위한 기준
    # totalParameter similarity: 재분석 결과의 유사도 점수 기준
    # body = {
    # "parameter":[{"similarity":"0.5"}],
    # "totalParameter":[{"threshold":"0.6","similarity":"0.6"}]
    # }
    if algorithm_id == '1':
        body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))
        result_intents = federated_engine.get_federated_intent(datahubs, query, body)
    else:
        result_intents = federated_engine.get_federated_intent(datahubs, query)

    result_map = {"result": result_intents}

    return json.dumps(result_map, ensure_ascii=False)


# desc: 통합 사용자 의도 분석 결과 중 선택한 결과를 기반으로 재검색 수행 -> 수정 필요
@app.route("/discovery/federated/intent-based/intent/search", methods=['POST'])
def federated_search_with_intent():
    # federated_engine.
    categories = request.args.get('categories')
    sites = request.args.get('sites')
    datahubs = federated_engine.make_search_hub_list(categories, sites)
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
        print('No keywords in.')
        return {'result': 'No keywords in.'}

    cache_dic = {}
    cache_dic['query'] = query
    cache_dic['portal']= datahubs
    cache_key = str(cache_dic)
    # 캐시가 있는 경우
    if pagination.is_cached(cache_key):
        result, total_len = pagination.get_data(cache_key, start, end)
    else:
        result_datasets = federated_engine.re_search(datahubs, query)
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

    result_map = {}

    query = request.args.get('searchValue')
    categories = request.args.get('categories')
    sites = request.args.get('sites')
    algorithm_id = request.args.get('algorithmId')

    datahubs = federated_engine.make_search_hub_list(categories, sites)

    if algorithm_id == '1':
        body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))

        # 캐시가 있는지 확인할 key_dictionary 생성
        cache_dic = {}
        cache_dic['query'] = query
        cache_dic['body'] = body
        cache_dic['portal'] = datahubs
        cache_key = str(cache_dic)

        if pagination.is_cached(cache_key):
            original_datasets = pagination.get_all_data(cache_key)

            # 불러온 경우 해당 데이터로 facet 객체 생성
            facet_obj = Facet(original_datasets)

        else:
            return json.dumps("잘못된 키가 들어왔습니다.", ensure_ascii=False)
    else:
        # 현재 알고리즘은 없음.
        #body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))

        cache_dic = {}
        cache_dic['query'] = query
        # default 값을 가진 body 생성
        body_value = {'parameter': [{'similarity': '0.5'}],
                      'totalParameter': [{'threshold': '0.55', 'similarity': '0.6'}]}
        cache_dic['body'] = body_value
        cache_dic['portal'] = datahubs
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
    categories = request.args.get('categories')
    sites = request.args.get('sites')
    algorithm_id = request.args.get('algorithmId')

    datahubs = federated_engine.make_search_hub_list(categories, sites)

    dataset_count = request.args.get('datasetCount')
    dataset_count = int(dataset_count)

    # 값이 오면 사용하고, 없는 경우 1을 defalut로 사용함.
    page_num = request.args.get('pageNum', default='1', type=str)
    page_num = int(page_num)

    # pagination을 위한 start, end 초기화
    start = dataset_count * (page_num - 1) + 1
    end = start + dataset_count - 1

    body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))
    # body에서 facet 관련 정보 우선 추출
    facets = body["facets"]
    # 이후 cache에 사용하기 위해 facets 부분은 제거하고 cache 키의 일부로 활용함.
    cache_body = body.copy()
    cache_body.pop('facets', None)



    # 현재 알고리즘이 없으므로 분기만 설정함.
    if algorithm_id is None:
        cache_dic = {}
        cache_dic['query'] = query
        # default 값을 가진 body 생성
        body_value = {'parameter': [{'similarity': '0.5'}],
                      'totalParameter': [{'threshold': '0.55', 'similarity': '0.6'}]}
        cache_dic['body'] = body_value
        cache_dic['portal'] = datahubs
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
        cache_dic['body'] = cache_body
        cache_dic['portal'] = datahubs
        cache_key = str(cache_dic)

        print(cache_key)

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
    result_map = {"totalPage": math.ceil(len(datasets)/ dataset_count),
                  "currentPage": page_num,
                  "filteredDatasets": result}

    return json.dumps(result_map, ensure_ascii=False)

# 통계 수치 반환 API
@app.route('/discovery/federated/intent-based/search/aggs')
def get_federated_search_aggs():
    query = request.args.get('searchValue')
    categories = request.args.get('categories')
    sites = request.args.get('sites')
    dataset_count = request.args.get('datasetCount', default=10, type=int)
    algorithm_id = request.args.get('algorithmId')
    dataset_count = int(dataset_count)

    datahubs = federated_engine.make_search_hub_list(categories, sites)

    print(datahubs)

    # algorithm_parameter는 다음과 같으며 3가지를 조정가능, 사용하지 않을 파라미터는 쓰지 않아도 동작함.
    # parameter similarity: 각각의 시맨틱 허브의 의도 분석 결과 유사도 점수 기준
    # totalParameter threshold: 각 허브 의도 통합 시 분석 결과에 포함시키기 위한 기준
    # totalParameter similarity: 재분석 결과의 유사도 점수 기준
    # body = {
    # "parameter":[{"similarity":"0.5"}],
    # "totalParameter":[{"threshold":"0.6","similarity":"0.6"}]
    # }
    if algorithm_id == '1':
        body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))

        # 캐시가 있는지 확인할 key_dictionary 생성
        cache_dic = {}
        cache_dic['query'] = query
        cache_dic['body'] = body
        cache_dic['portal'] = datahubs
        cache_key = str(cache_dic)

        # 캐시가 있는 경우
        if pagination.is_cached(cache_key):
            result = pagination.get_all_data(cache_key)
            aggs = federated_engine.get_federated_search_aggs(result)
        else:
            # 통합 검색 수행
            result_datasets = federated_engine.federated_search_ctrl(datahubs, query, body)
            pagination.insert_data(cache_key, result_datasets)
            result = pagination.get_all_data(cache_key)
            aggs = federated_engine.get_federated_search_aggs(result)
    else:
        # 캐시가 있는지 확인할 key_dictionary 생성
        cache_dic = {}
        cache_dic['query'] = query
        # default 값을 가진 body 생성
        body_value = {'parameter': [{'similarity': '0.5'}],
                      'totalParameter': [{'threshold': '0.55', 'similarity': '0.6'}]}
        cache_dic['body'] = body_value
        cache_dic['portal'] = datahubs
        cache_key = str(cache_dic)
        # 캐시가 있는 경우
        if pagination.is_cached(cache_key):
            print(cache_key)
            result = pagination.get_all_data(cache_key)
            aggs = federated_engine.get_federated_search_aggs(result)
        # 캐시가 없는 경우
        else:
            result_datasets = federated_engine.federated_search_ctrl(datahubs, query)
            pagination.insert_data(cache_key, result_datasets)
            result = pagination.get_all_data(cache_key)
            aggs = federated_engine.get_federated_search_aggs(result)

    result = json.dumps(aggs, ensure_ascii=False).encode('utf-8')
    res = make_response(result)

    return res


# desc: 통합 사용자 의도 분석 결과 중 선택한 결과를 기반으로 재검색을 수행한 결과에 대한 통계 반환 API
@app.route("/discovery/federated/intent-based/intent/search/aggs", methods=['POST'])
def get_federated_search_with_intent_aggs():
    # federated_engine.
    categories = request.args.get('categories')
    sites = request.args.get('sites')
    datahubs = federated_engine.make_search_hub_list(categories, sites)
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
        print('No keywords in.')
        return {'result': 'No keywords in.'}

    cache_dic = {}
    cache_dic['query'] = query
    cache_dic['portal']= datahubs
    cache_key = str(cache_dic)
    # 캐시가 있는 경우
    if pagination.is_cached(cache_key):
        result = pagination.get_all_data(cache_key)
        aggs = federated_engine.get_federated_search_aggs(result)
    else:
        result_datasets = federated_engine.re_search(datahubs, query)
        pagination.insert_data(cache_key, result_datasets)
        result = pagination.get_all_data(cache_key)
        aggs = federated_engine.get_federated_search_aggs(result)

    result = json.dumps(aggs, ensure_ascii=False).encode('utf-8')
    res = make_response(result)

    return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7700, debug=True)
