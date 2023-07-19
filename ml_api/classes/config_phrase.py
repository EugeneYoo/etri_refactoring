import os
import logging

# desc: hub 객체를 초기화 하기 위한 config 정보
# input
# model: 데이터 허브의 의도 분석 모델(word2vec bin)파일 - 추가 모델은 model 폴더 참조
# elasticsearch: hub dataset이 저장된 elasticsearch 서버 주소
# hub_index_name: hub의 dataset이 저장된 es index name
args = \
    {
        'development': {
            'model': r'./model/phrase/ko_combine.bin',
            'raw_model': './model/ko_new.bin',
            'query_model' : r'./model/query_txt.bin',
            #'hub_index_name': 'total_datasets2',
            'hub_index_name': 'total_datasets2',
            'elasticsearch': '163.180.116.87:9200',
            #'elasticsearch': '14.37.210.100:9200',
            #'elasticsearch' : '14.37.210.100:9200',
            # 'hub_index_name': 'sodas+-default-public.asset',
            # 'elasticsearch': 'https://admin:admin@129.254.76.168:9200',
            'use_ssl': False,  # True,
            'verify_certs': False,
            'redis_server': '163.180.116.87',
            'redis_port': 6379,
            'redis_db_number': 2,
            'redis_expire_time': 300000,  # (seconds)
            'expand_limit': 5,  # word2vec으로 찾을 유사 단어 갯수
            'logging_server': '163.180.116.70',
            'logging_server_dbname': 'postgres',
            'logging_server_user_id': 'postgres',
            'logging_server_password': '1234',
            'logging_server_port': 5432,
            'python_logging_lvl': logging.WARN
        },
        'production': {
            'model': os.environ['model'] if 'model' in os.environ else './model/ko_latest_IT과학.bin',
            'elasticsearch': os.environ[
                'elasticsearch'] if 'elasticsearch' in os.environ else 'https://admin:admin@129.254.76.168:9200',
            'hub_index_name': os.environ[
                'hub_index_name'] if 'hub_index_name' in os.environ else 'sodas+-default-public.asset',
            'use_ssl': os.environ['use_ssl'] if 'use_ssl' in os.environ else True,
            'verify_certs': os.environ['verify_certs'] if 'verify_certs' in os.environ else False,
            'redis_server': 'localhost',
            'redis_port': 6379,
            'redis_db_number': 0,
            'redis_expire_time': 300,  # (seconds)
        }
    }
