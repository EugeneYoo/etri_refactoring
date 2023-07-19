import json
import os
import time

from flask import Flask
from flask import redirect, flash, url_for, render_template, request, send_file
from flask import make_response
from flask_cors import CORS

from werkzeug.utils import secure_filename


from classes.datahub_phrase import DataHub
from classes.pagination_phrase import Pagination
from classes.config_phrase import args

from train_phrase_w2v import retrainScriptForNewESDataNoPhrase as retrain

from classes.logdb import LogDB

import classes.ranking as rankService
from classes.ranker import Ranker

from operator import itemgetter

import classes.trainQDS as TQ

ranker = Ranker.get_instance()

import requests

from classes.twoin_func import *
from classes.error_handler import error_handle

# from waitress import serve

from gensim.models import KeyedVectors
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
app.config['threaded'] = False

CORS(app)
error_handle(app)

# api를 사용할 객체 선언
hub = DataHub(hub_index_name=config['hub_index_name'], elasticsearch=config['elasticsearch'], use_ssl=config['use_ssl'],
                verify_certs=config['verify_certs'], model=config['model'])

# 시맨틱 검색 페이지를 pagination 객체 선언
pagination = Pagination()

# 모델 아이디 관리를 위한 global
w2vmodel_id = 4
remodel_id = 1


# api , 데이터셋 길이가 0인경우 반환하면서 메시지 표기
@app.route("/train/test/<flag>/<sd>/<ed>")
def do_train_process(flag, sd, ed):
    # start_time을 체크
    start_time = time.time()

    # 2022-11-08
    result = retrain.get_data_with_date(sd, ed)
    # result= get_data_with_date()
    result = retrain.extract_txt(result)
    result = retrain.preprocessing(result)

    if (flag == 'retrain'):
        result = retrain.train_model(model_path=config['model'], datasets=result)
    elif (flag == 'new'):
        result = retrain.train_model(model_path=config['raw_model'], datasets=result)
    result['time'] = "소요시간: ---{}s seconds---".format(time.time() - start_time)

    return json.dumps(result, ensure_ascii=False)


@app.route('/log/train/query')
def train_query_log_data():
    import codecs

    file = 'query_txt'

    print("Making sentences as list...")
    datasets = []

    with codecs.open(f'{file}.txt', 'r', 'utf-8') as fin:
        while 1:
            line = fin.readline()
            if not line: break

            words = line.split()
            datasets.append(words)
        print("data lodding complete")

    print(f"modeldldeledl{config['model']}")
    dic = retrain.train_model(datasets, model_path=config['model'])
    hub.intent_analyzer.pretrained_model = KeyedVectors.load_word2vec_format(r'C:\Users\hi\Desktop\khu_2\khu\Datahub\query_txt.bin', binary=True,
                                                                            unicode_errors='ignore')
    hub.intent_analyzer.model_path = r'C:\Users\hi\Desktop\khu_2\khu\Datahub\query_txt.bin'

    return "train complete"

@app.route('/discovery/datasets/id/<id>')
def get_id(id):
    body= \
        {
            "query": {
                "match": {
                    "uid": id
                }
            }
        }
    results = hub.es.search(index= hub.hub_index_name, body=body)
    results = results['hits']['hits'][0]['_source']
    res = {}
    res['title']= results['title']
    res['desc']= results['desc']
    res['uid']= results['uid']
    result={"results": res}
    return result


@app.route('/log/train/click')
def train_click_log_datas():
    print("접근")
    DB = LogDB()
    # a: click id
    # b: datasetlist -=> 검색으로 화면에 출력된 것
    # C: query string
    click_id, dataset_list, query_string = DB.get_train_data()

    print('get data complete')
    for i in range(0, len(click_id)):
        # rerank_result = rankService.getRank(c[i], b[i])

        tmp = ranker.setDatasetVector(dataset_list[i])
        print(f'{tmp},  set dset vec complete')
        ranker.setResultDataset(tmp)
        print(f'{ranker.getResultDataset()},  set result datasets')
        negatives = rankService.negativeSampling(click_id[i])
        print(f'{negatives}, ns complete')
        positive = ranker.getClickedDatasetVector()
        print(f"{positive}, get clickdataset vect")
        ranker.setQueryVector(query_string[i])
        print(f"{ranker.getQueryVector()}, query vec comp")
        queryVector = ranker.getQueryVector()

        result = rankService.concatenateQueryDocumentVector(negatives, positive, queryVector)
        print(f'{result}, concat Doc vec')
        resultDf = rankService.makeDataFrame(result)
        print(f'{resultDf}, result df ')
        postDf = ranker.getQDSRawDataset()
        print(f'{postDf}, getQDSRawDataset()')
        if postDf is not None:
            resultDf = rankService.concatenateDataframe(resultDf, postDf)
        resultDf.to_csv("./dataset/QDSRawDataset.csv", header=True, encoding='utf-8-sig', index=None)

    TQ.train_QDS()

    return "train complete"

@app.route('/model/original')
def change_model_ori():
    hub.intent_analyzer.pretrained_model = KeyedVectors.load_word2vec_format(r'C:\Users\hi\Desktop\khu_2\khu\Datahub\model\phrase\ko_combine.bin', binary=True,
                                                                            unicode_errors='ignore')
    hub.intent_analyzer.model_path = r'C:\Users\hi\Desktop\khu_2\khu\Datahub\model\phrase\ko_combine.bin'

    return "오리지널 모델"


@app.route('/model/query')
def change_model_query():
    hub.intent_analyzer.pretrained_model = KeyedVectors.load_word2vec_format('./query_txt.bin', binary=True, unicode_errors='ignore')
    hub.intent_analyzer.model_path = './query_txt.bin'

    return "쿼리 학습 모델"

@app.route('/model/query/<id>')
def change_model_query_with_id(id):
    path = f'./data/w2v_{id}.bin'
    hub.intent_analyzer.pretrained_model = KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors='ignore')
    hub.intent_analyzer.model_path = path
    return "학습 모델"+ path


@app.route("/model/data/upload", methods=['GET', 'POST'])
def upload_model_train_data():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_model_train_data', name=filename))
        return '파일이 저장되었습니다.'
    else:
        return render_template('file_upload.html')

@app.route("/model/get")
def download_model_file():
    flag = request.args.get('flag')
    model_id = str(request.args.get('modelId'))
    if flag =="w2v":
        file = "w2v_"+model_id+".bin"
    elif flag =="rerank":
        file = "rerank_"+model_id+".pt"

    print(file)

    return send_file(f'data/{file}')
    #return send_file(os.path.join(filepath, filename), as_attachment=True)
    #as_attachment – Indicate to a browser that it should offer to save the file instead of displaying it.
    #download_name – The default name browsers will use when saving the file. Defaults to the passed file name.

@app.route("/model/w2v/train", methods=['POST'])
def train_w2v_model():
    global w2vmodel_id

    try:
        model_name = "w2v_"+str(w2vmodel_id)+".bin"
        w2vmodel_id+=1
        body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))
        trainData = int(request.args.get('trainData'))
        modelParams = body['modelParams']
        trainData = get_train_data("w2v", trainData)
        start = time.time()
        retrain.make_wordvectors(trainData, model_name, modelParams)
        end = time.time()

        model_path = "data/" + str(model_name)

        hub.make_new_analyzer_object(model_path)
        pagination.flush_old_data()

        dic ={}
        dic['time'] = end - start
        dic['modelId'] = w2vmodel_id-1
    except:
        return make_response("서버에서 요청을 처리하지 못했습니다.", 500)

    return make_response(dic)


@app.route("/model/w2v/retrain")
def retrain_w2v():
    model_id = request.args.get('modelId')
    model_name = "data/w2v_" + str(model_id) + ".bin"
    trainData = int(request.args.get('trainData'))
    trainData = get_train_data("w2v", trainData)
    start = time.time()
    retrain.train_model(trainData, model_name)
    end = time.time()

    model_path = model_name
    hub.make_new_analyzer_object(model_path)
    pagination.flush_old_data()

    dic = {}
    dic['time'] = end- start
    dic['modelId'] = model_id

    return dic

@app.route("/model/rerank/train")
def train_rerank():    
    global remodel_id
    model_id = "rerank_" + str(remodel_id) + ".pt"
    remodel_id += 1
    trainData = int(request.args.get('trainData'))
    start = time.time()
    trainData = get_train_data("qds", trainData)
    print("rerank_data load complete")
    TQ.train_QDS_with_Name(trainData, model_id)
    print("rerank_data train complete")
    end = time.time()

    dic = {}
    dic['time'] = end - start
    dic['modelId'] = remodel_id - 1
    return dic


@app.route("/model/rerank/retrain")
def retrain_rerank():
    model_id = request.args.get('modelId')
    model_id = "rerank_" + str(model_id) + ".pt"
    trainData = int(request.args.get('trainData'))
    start = time.time()
    trainData = get_train_data("qds", trainData)
    TQ.train_QDS_with_Name(trainData, model_id)
    end = time.time()

    dic = {}
    dic['time'] = end - start
    dic['modelId'] = remodel_id - 1
    return dic


@app.route("/discovery/semantic/intent-based/search/metrics", methods=["POST"])
def metric_api():
    query = request.args.get('searchValue')
    print("query: ",query)
    dataset_count = request.args.get('datasetCount', default=10, type=int)
    algorithm_id = request.args.get('algorithmId')

    dataset_count = int(dataset_count)

    body = json.loads(request.get_data().decode('utf-8').encode('utf-8'))
    groundTruth = body.pop('groundTruth')

    # 추가된 부분
    # 값이 오면 사용하고, 없는 경우 1을 defalut로 사용함.
    page_num = request.args.get('pageNum', default='1', type=str)
    page_num = int(page_num)

    start = dataset_count * (page_num - 1) + 1
    end = start + dataset_count - 1

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
            result = pagination.get_all_data(cache_key)
            total_len = len(result)
        else:
            # 캐시가 없는 경우
            # str list sub element join with and, str list element join with or
            query = query.split()
            print(query)
            modified_query_list = []
            sub_body = " AND ".join(query)
            sub_body = "(" + sub_body + ")"

            search_result = hub.search(sub_body)
            pagination.insert_data(cache_key, search_result)
            result = pagination.get_all_data(cache_key)
            total_len = len(result)

    # threshold 알고리즘 파라미터를 적용한 검색
    elif algorithm_id == '1':
        body['parameter'] = body.pop('algorithmParam')

        # 캐시가 있는지 확인할 key_dictionary 생성
        cache_dic = {}
        cache_dic['query'] = query
        cache_dic['body'] = body
        cache_key = str(cache_dic)
        # 캐시가 있는 경우
        if pagination.is_cached(cache_key):
            result = pagination.get_all_data(cache_key)
            total_len = len(result)
        # 캐시가 없는 경우
        else:
            threshold = float(body['parameter'][0]['similarity'])
            dataset = hub.semantic_search(query, algorithm_id=algorithm_id, algorithm_parameter=threshold)
            pagination.insert_data(cache_key, dataset)

            result = pagination.get_all_data(cache_key)
            total_len = len(result)
            resultids = []

    elif algorithm_id == "2":
        # rerank 임시 적용
        body['parameter'] = body.pop('algorithmParam')
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
            slice_data = result_all[:30]
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
            result = pagination.get_all_data(cache_key)
            total_len = len(result)
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

            result = pagination.get_all_data(cache_key)
            total_len = len(result)

    recall, precision = calc_metric(result, total_len,groundTruth)

    # top10 = result[:10]

    dic={}
    dic['totalDataset'] = total_len
    dic['precisionAt10'] = precision
    dic['recall'] = recall
    # dic['Top10'] = top10

    return json.dumps(dic, ensure_ascii=False)


@app.route('/')
def main_page():
    print("hit main page")
    return {"success": True, "msg": "api 주소를 입력해주세요"}


if __name__ == "__main__":
    print("activate!")
    app.run(host='0.0.0.0', port=8600, debug=True)
    #serve(app, host='0.0.0.0', port=8600, url_prefix="/ml")
    print("deactivate!")

