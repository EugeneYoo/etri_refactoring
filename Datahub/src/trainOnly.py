import codecs
import logging
import os
from gensim.models import Word2Vec, KeyedVectors
from classes.confing_phrase import args

env = 'development' if not 'APP_ENV' in os.environ else os.environ['APP_ENV']
config = args[env]

logging.basicConfig(level=config['python_logging_lvl'])

config1={'re_save': 'query_txt.bin'}
file = 'query_txt'

logging.info("Making sentences as list...")

datasets = []

with codecs.open(f'{file}.txt', 'r', 'utf-8') as fin:
    while 1:
        line = fin.readline()
        if not line: break

        words = line.split()
        datasets.append(words)
    logging.info("data lodding complete")
def check_shape(model): # word2vec 모델 크기를 출력하기 위한 함수
    logging.info(f"model shape: {model.wv.vectors.shape}")
    ms = model.wv.vectors.shape
    return ms



# 학습 수행하기(전처리된 리스트 데이터 묶음 한번에 받자)
def train_model(datasets, model_path, binary=True):
    model_2 = Word2Vec(size=200, min_count=1)  # 사전 학습 모델 사이즈가 200이므로 200 고정
    model_2.build_vocab(datasets)

    total_examples = model_2.corpus_count
    model = KeyedVectors.load_word2vec_format(model_path, binary=binary)
    model_2.build_vocab([list(model.vocab.keys())], update=binary)  # build vocab에서 시간이 걸림

    # 기본적으로는 외부에서 가져온 vector에 어떤 수정도 진행되지 못하도록 lock이 걸려 있음.
    # 따라서, 만약 추가로 vector를 업데이트하고 싶다면, lockf=1.0으로 argument를 넘겨주는 것이 필요함
    # 다음처럼 이미 만들어진 word2vec model에 외부 파일로에 저장된 keyed_vector의 weight를 덮어 씌워주는 것.
    model_2.intersect_word2vec_format(model_path, binary=binary, lockf=1.0)
    logging.info("학습 시작")
    model_2.train(datasets, total_examples=total_examples, epochs=model_2.epochs)

    model_2.wv.save_word2vec_format(config1['re_save'], binary=True)
    logging.info("저장 완료")

    new_vocab = []
    for k in model_2.wv.vocab:
        result = model.wv.vocab.get(k, "0")
        if result == "0":
            new_vocab.append(k)



    logging.info("new_model.bin's train success!!")

    dic= {}


    dic['before model'] = check_shape(KeyedVectors.load_word2vec_format(model_path, binary=True))
    dic['train model'] = check_shape(KeyedVectors.load_word2vec_format(config1['re_save'], binary=True))
    dic['new_vocab'] = new_vocab

    return dic


"""

from classes.queryintentanalyzer import QueryIntentAnalyzer
from pprint import pprint as pp

word = '쿠데타'


analyzer = QueryIntentAnalyzer('model/ko_trans_keyedvector.bin')

result1 = analyzer.query_intent_analysis_with_score(word)


analyzer1 = QueryIntentAnalyzer('model/ko_new.bin')
        
result2 = analyzer1.query_intent_analysis_with_score(word)

pp(result1)
pp(result2)


reformat.py

다운받은 사전학습모델(gensim full model)을 추가학습 코드에 사용 가능하도록 keyedVector로 다시 저장함.

gensim full model => Word2Vec 파일 자체 저장
gensim keyedVectors => 추가로 벡터를 업데이트 할 수 없지만 확장성이 높고 모델 상태를 저장하지 않으므로 가벼움


import gensim


pretrained_model = gensim.models.Word2Vec.load('ko_trans_stop.bin')
pretrained_model.wv.save_word2vec_format('ko_trans_keyedvector.bin', binary=True)

# save 한 keyedVector 가 제대로 로딩이 되는지 확인
gensim.models.KeyedVectors.load_word2vec_format('ko_trans_keyedvector.bin', binary=True, unicode_errors='ignore')

"""