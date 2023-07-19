import regex
from datetime import datetime
import logging
import pickle
import time
import gensim  # In case you have difficulties installing gensim, you need to consider installing conda.
import codecs

from elasticsearch import Elasticsearch
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from konlpy.tag import Mecab
analyzer = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")  # Mecab 형태소 분석기 로딩

# desc: hub 객체를 초기화 하기 위한 config dictionary
# input
# model: 데이터 허브의 의도 분석 모델(word2vec bin)파일 - 추가 모델은 model 폴더 참조
# elasticsearch: hub dataset이 저장된 elasticsearch 서버 주소
# hub_index_name: hub의 dataset이 저장된 es index name
config = {
    'model': r'C:\Users\hi\Desktop\khu_2\khu\Datahub\model\before_train_data_model.bin',
    'elasticsearch': 'http://14.37.211.69:9200',
    'hub_index_name': 'total_datasets2',
    # 새로 저장할 경로 (수정 필수)
    're_save': r'C:\Users\hi\Desktop\khu_2\khu\Datahub\model\True_full_model.bin'
}



es = Elasticsearch(config['elasticsearch'])

def make_sort_query(sd, ed, md, id):
    print("md, iddd: ", md, id)
    query_body = \
        {
            "sort": {
                "modified_date": "asc",
                "uid": "asc"
            },
            "search_after" : [ md, id],
            "query": {
                "range": {
                    "modified_date": {
                        "gte": sd,
                        "lte": ed,
                        "format": "yyyy-MM-dd||yyyy-MM-dd"
                    }
                }
            }
        }
    return query_body


# 데이터 es에서 불러오는 함수 date 에따라 해당하는 데이터 불러와서 제목, 설명을 추출함
def get_data_with_date(start_date='1970-01-01', end_date=datetime.now().strftime('%Y-%m-%d')):
    dataset = []
    query_body= \
        {
            "sort":{
                "modified_date": "asc",
                "uid": "asc"
            },
            "query": {
                "range": {
                    "modified_date": {
                        "gte": start_date,
                        "lte": end_date,
                        "format": "yyyy-MM-dd||yyyy-MM-dd"
                    }
                }
            }
        }
    result= es.search(index=config['hub_index_name'], body=query_body, size=5)
    # 검색 결과 있는지 확인
    while len(result['hits']["hits"]) != 0:
        search_result = []
        for data in result['hits']['hits']:
            datasets = data['_source']
            search_result.append(datasets)
        # 값을 전부 결과 배열에 넣음
        dataset.extend(search_result)
        # 마지막 sort 값 확인
        sort_modi_date, sort_uid = result['hits']["hits"][-1]['sort']
        new_query_body = make_sort_query(start_date, end_date, sort_modi_date, sort_uid)
        # 새로운 쿼리로 다시 이후 결과를 받아 반복
        result=es.search(index=config['hub_index_name'], body= new_query_body)

    return dataset

# 데이터셋 리스트에서 제목, 설명만 따옴
def extract_txt(datasets):
    if len(datasets) == 0:
        return "데이터셋이 없습니다."
    result = []
    for data in datasets:
        result.append(data['desc'])
        result.append(data['title'])

    return result

def word_processing(datasets):
    result = []
    for data in datasets:
        if data != None:
            #print("데이터", data)
            tmp_data = analyzer.morphs(data)
            if(len(tmp_data)> 10):
                pass
            else:
                result.append(tmp_data)
    return result

# 불러온 데이터셋의 전처리 함수(불용어 및 형태소 분석)
def preprocessing(datasets):
    if len(datasets) == 0:
        return "데이터셋이 없습니다."
    
    result = []
    for data in datasets:
        txt = data.strip()
        subtxt = regex.sub("[^ \r\n\p{Hangul}.?!]", " ", txt)
        # 문장 단위 스플릿
        sents = regex.split("([.?!])?[\n]+|[.?!] ", subtxt)
        result.extend(sents)
    # 형태소 문장 적용
    w_result = word_processing(result)

    return w_result

def check_shape(model): # word2vec 모델 크기를 출력하기 위한 함수
    print("model shape: ", model.wv.vectors.shape)
    ms = model.wv.vectors.shape
    return ms

# 학습 수행하기(전처리된 리스트 데이터 묶음 한번에 받자)
def train_model(datasets, model_path='ko_new.bin', binary=True):
    model_2 = Word2Vec(size=200, min_count=1)  # 사전 학습 모델 사이즈가 200이므로 200 고정
    model_2.build_vocab(datasets)
    total_examples = model_2.corpus_count
    model = KeyedVectors.load_word2vec_format(model_path, binary=binary)
    model_2.build_vocab([list(model.vocab.keys())], update=binary)  # build vocab에서 시간이 걸림

    # 기본적으로는 외부에서 가져온 vector에 어떤 수정도 진행되지 못하도록 lock이 걸려 있음.
    # 따라서, 만약 추가로 vector를 업데이트하고 싶다면, lockf=1.0으로 argument를 넘겨주는 것이 필요함
    # 다음처럼 이미 만들어진 word2vec model에 외부 파일로에 저장된 keyed_vector의 weight를 덮어 씌워주는 것.
    model_2.intersect_word2vec_format(model_path, binary=binary, lockf=1.0)
    print("학습 시작")
    model_2.train(datasets, total_examples=total_examples, epochs=model_2.epochs)

    model_2.wv.save_word2vec_format(config['re_save'], binary=True)
    print("저장 완료")

    new_vocab = []
    for k in model_2.wv.vocab:
        result = model.wv.vocab.get(k, "0")
        if result == "0":
            new_vocab.append(k)



    logging.info("new_model.bin's train success!!")

    dic= {}


    dic['before model'] = check_shape(KeyedVectors.load_word2vec_format(model_path, binary=True))
    dic['train model'] = check_shape(KeyedVectors.load_word2vec_format(config['re_save'], binary=True))
    dic['new_vocab'] = new_vocab

    return dic

# 최초 학습 함수
def make_wordvectors(data, model_id, param):
    
    vector_size = param['vector_size'] if 'vector_size' in param.keys() else 200
    negative = param['negative'] if 'negative' in param.keys() else 5
    window = param['window'] if 'window' in param.keys() else 5
    min_count = param['min_count'] if 'min_count' in param.keys() else 1
    logging.info("Making sentences as list...")
    sents = []
    start =  time.time()
    with codecs.open('data/{}'.format(data), 'r', 'utf-8') as fin:
        while 1:
            line = fin.readline()
            if not line: break

            words = line.split()
            sents.append(words)
    end = time.time()
    logging.info(f"spend time: {end-start}")
    logging.info("Making word vectors...")
    model = gensim.models.Word2Vec(sents, size=vector_size, min_count=min_count,
                                    negative=negative,
                                    window=window)
    logging.info("Done!!")
    model.wv.save_word2vec_format('data/{}'.format(model_id), binary=True)
    logging.info("save keyedvector")

    logging.info("save setting")

    # save data
    setting = {}
    setting['vector_size'] = vector_size
    setting['min_count'] = min_count
    setting['window'] = window
    setting['negative'] = negative

    logging.info(f"modelid: {model_id}")
    with open(f'{model_id}.pickle', 'wb') as fw:
        pickle.dump(setting, fw)

    # load data
    #with open('user.pickle', 'rb') as fr:
    #    user_loaded = pickle.load(fr)

