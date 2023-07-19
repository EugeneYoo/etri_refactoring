import os
import logging
# desc: 통합 검색 엔진 객체를 초기화 하기 위한 config 정보
# input
# integrate_model: 통합 검색 앤진의 통합 모델(word2vec bin)파일
# pgsql_host: 통합 검색 데이터 허브 메타데이터 정보를 저장한 pgsql host
# pgsql_dbname: 통합 검색 데이터 허브 메타데이터 정보를 저장한 pgsql dbname
# pgsql_user: pgsql에 접근하기 위한 user 이름
# pgsql_password: pgsql에 접근하기 위한 user pw
# pgsql_port: pgsql 포트 넘버
# federated_member_hub_table: 통합 검색을 수행할 수 있는 맴버 포털들의 정보가 저장된 pgsql table name
# max_worker: 통합 검색 엔진에서 동시 request에 사용할 Thread worker의 수


args = \
    {
        'development': {
            'integrate_model': './model/ko_latest_통합모델.bin',
            'pgsql_host': '163.180.116.70',
            'pgsql_dbname': 'postgres',
            'pgsql_user': 'postgres',
            'pgsql_password': '1234',
            'pgsql_port': 5432,
            'federated_member_hub_table': 'hub_info_docker',
            'max_worker': 4,
            'redis_server': '163.180.116.70',
            'redis_port': 6379,
            'redis_db_number': 0,
            'redis_expire_time': 300000,  # (seconds)
            'python_logging_lvl': logging.DEBUG
        },
        'production': {
            'integrate_model': os.environ['integrate_model'] if 'integrate_model' in os.environ \
                else './model/ko_latest_통합모델.bin',
            'pgsql_host': os.environ['pgsql_host'] if 'pgsql_host' in os.environ else '163.180.116.70',
            'pgsql_dbname': os.environ['pgsql_dbname'] if 'pgsql_dbname' in os.environ else 'postgres',
            'pgsql_user': os.environ['pgsql_user'] if 'pgsql_user' in os.environ else 'postgres',
            'pgsql_password': os.environ['pgsql_password'] if 'pgsql_password' in os.environ else '1234',
            'pgsql_port': int(os.environ['pgsql_port']) if 'pgsql_port' in os.environ else 5432,
            'federated_member_hub_table': 'hub_info_docker',
            'max_worker': int(os.environ['max_worker']) if 'max_worker' in os.environ else 4,
            'redis_server': 'localhost',
            'redis_port': 6379,
            'redis_db_number': 0,
            'redis_expire_time': 300,  # (seconds)
            'testing': os.environ['Testing'] if 'Testing' in os.environ else 'no env'
        }
    }
