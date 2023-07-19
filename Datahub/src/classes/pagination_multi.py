import json
import os
import redis
import threading
import logging
from .config_phrase import args

# Variables receiving environment settings
env = 'development' if not 'APP_ENV' in os.environ else os.environ['APP_ENV']
config = args[env]

# redis expire time
expire_time = config['redis_expire_time']


import redis
from redis.lock import Lock
class Pagination:
    def __init__(self):
        self.redis_conn = redis.StrictRedis(host=config['redis_server'], port=config['redis_port'], db=config['redis_db_number'])
        self.redis_lock = Lock(self.redis_conn, 'pagination_lock')

    def insert_data(self, key, data):
        if type(key) is not str:
            key = str(key)

        data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        # Acquire a lock
        with self.redis_lock:
            with self.redis_conn.pipeline() as pipe:
                if self.redis_conn.exists(key) == False:
                    pipe.setex(key, expire_time, data)
                    pipe.execute()
                    print("cache saving")
                    return True
                else:
                    logging.info("캐싱된 값이 존재함")
                    return False

    def get_data(self, key, start, end):
        """
        key를 통해 캐싱된 데이터셋을 불러와서 페이지 네이션을 수행한 데이터를 넘겨준다.
        해당 데이터의 전체 크기도 함께 전송함(total_page 계산).

        :param key: (int) 검색 결과를 불러올 key
        :param start: (int) pagination 시작 페이지
        :param end: (int) pagination 종료 페이지
        :return: (list in dict) pagination이 적용된 데이터셋 리스트,
                 data_length: 저장된 데이터 길이
        """
        if type(key) is not str:
            key = str(key)

        # Acquire a lock
        with self.redis_lock:
            with self.redis_conn.pipeline() as pipe:
                if self.redis_conn.exists(key):
                    pipe.get(key)
                    data = pipe.execute()[0]
                    logging.info(data)
                else:
                    logging.info("저장되지 않은 캐시")
                    print("no saving")
                    return []

            data = json.loads(data.decode('utf-8'))
            result = []
            if len(data) < end:
                end = len(data)
            for i in range(start - 1, end):
                result.append(data[i])
            return result, len(data)

    def is_cached(self, key):
        if self.redis_conn.exists(key) == 1:
            return True
        else:
            return False

    def flush_all(self):
        try:
            with self.redis_lock:
                logging.info("flush all")
                self.redis_conn.flushall()
        except redis.exceptions.LockNotOwnedError:
            logging.info("Lock not owned, cannot flush all")


    def update_expire_time(self, key, expire_time=expire_time):
        with self.redis_lock:
            with self.redis_conn.pipeline() as pipe:
                pipe.expire(key, expire_time)
                pipe.execute()

    def modify_data(self, key, new_data):
        with self.redis_lock:
            with self.redis_conn.pipeline() as pipe:
                pipe.set(key, new_data)
                pipe.execute()

    def get_all_data(self, cache_key):
        """
        facet class에게 facet key, value를 추출하기 위한 전체 데이터셋을 전달해주는 함수

        :param query_id: (int) 검색 결과를 불러올 query_id
        :return: (list in dict) query_id에 저장된 전체 데이터셋 리스트
        """
        # 잘못된 쿼리 아이디가 들어올 경우의 예외처리
        if not self.redis_conn.exists(cache_key):
            raise ValueError('invalid cache_key')

        # Acquire a lock
        with self.redis_lock:
            with self.redis_conn.pipeline() as pipe:
                pipe.expire(cache_key, expire_time)
                pipe.get(cache_key)
                data = pipe.execute()[0]
                return data
        return -1

