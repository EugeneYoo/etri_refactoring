import os
import regex
import codecs

import pickle

from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from konlpy.tag import Okt
#analyzer = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")  # Mecab 형태소 분석기 로딩

import logging
from Datahub.src.classes.config_phrase import args
env = 'development' if not 'APP_ENV' in os.environ else os.environ['APP_ENV']
config = args[env]

logging.basicConfig(level=config['python_logging_lvl'])

analyzer = Okt()

# desc: hub 객체를 초기화 하기 위한 config dictionary
# input
# model: 데이터 허브의 의도 분석 모델(word2vec bin)파일 - 추가 모델은 model 폴더 참조
# elasticsearch: hub dataset이 저장된 elasticsearch 서버 주소
# hub_index_name: hub의 dataset이 저장된 es index name

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
            tmp_data = analyzer.phrases(data)
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
    logging.info(f"model shape: {model.wv.vectors.shape}")
    ms = model.wv.vectors.shape
    return ms

# 학습 수행하기(전처리된 리스트 데이터 묶음 한번에 받자)
def train_model(data, model_path='ko_new.bin', binary=True):
    model_id = model_path.split('_')[1]
    model_id = model_id.split('.')[0]


    with open(f'w2v_{model_id}.bin.pickle', 'rb') as fr:
        setting = pickle.load(fr)

    vector_size = setting['vector_size'] if 'vector_size' in setting.keys() else 200
    negative = setting['negative'] if 'negative' in setting.keys() else 5
    window = setting['window'] if 'window' in setting.keys() else 5
    min_count = setting['min_count'] if 'min_count' in setting.keys() else 1

    model_2 = Word2Vec(size=vector_size, min_count=min_count,
                                    negative=negative,
                                    window=window)  # 사전 학습 모델 사이즈가 200이므로 200 고정
    sents = []
    with codecs.open('data/{}'.format(data), 'r', 'utf-8') as fin:
        while 1:
            line = fin.readline()
            if not line: break

            words = line.split()
            sents.append(words)

    logging.info("build vocab...")
    model_2.build_vocab(sents)
    total_examples = model_2.corpus_count
    model = KeyedVectors.load_word2vec_format(model_path, binary=binary)
    model_2.build_vocab([list(model.vocab.keys())], update=binary)  # build vocab에서 시간이 걸림

    # 기본적으로는 외부에서 가져온 vector에 어떤 수정도 진행되지 못하도록 lock이 걸려 있음.
    # 따라서, 만약 추가로 vector를 업데이트하고 싶다면, lockf=1.0으로 argument를 넘겨주는 것이 필요함
    # 다음처럼 이미 만들어진 word2vec model에 외부 파일로에 저장된 keyed_vector의 weight를 덮어 씌워주는 것.
    model_2.intersect_word2vec_format(model_path, binary=binary, lockf=1.0)
    logging.info("학습 시작")
    model_2.train(sents, total_examples=total_examples, epochs=model_2.epochs)

    logging.info('new_model' + "'s train success!!")

    new_vocab = []
    for k in model_2.wv.vocab:
        result = model.wv.vocab.get(k, "0")
        if result == "0":
            new_vocab.append(k)

    model_2.wv.save_word2vec_format(model_path, binary=True)
    logging.info("저장 완료")

    dic= {}
    dic['new_vocab'] = new_vocab

    return dic

# 최초 학습 함수
def make_wordvectors(data, model_id, param):
    import gensim  # In case you have difficulties installing gensim, you need to consider installing conda.

    vector_size = param['vector_size'] if 'vector_size' in param.keys() else 200
    negative = param['negative'] if 'negative' in param.keys() else 5
    window = param['window'] if 'window' in param.keys() else 5
    min_count = param['min_count'] if 'min_count' in param.keys() else 1
    logging.info("Making sentences as list...")
    sents = []
    import time
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