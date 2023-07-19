import json
import os
import redis
from ..src.classes.config_phrase import args


# 환경 설정 받아오는 변수
env = 'development' if not 'APP_ENV' in os.environ else os.environ['APP_ENV']
config = args[env]

# redis expire time
expire_time = config['redis_expire_time']


class Pagination:
    def __init__(self):
        """
        # Pagination 클래스

        params
        ------
        redis_conn: 캐시를 저장할 redis connect

        query_count: (int) cache_value_table에 저장된 데이터셋 index 역할 수행.\n

        """
        self.redis_conn = redis.StrictRedis(host=config['redis_server'], port=config['redis_port'], db=config['redis_db_number'])
        self.query_count = 1

    def get_cached_data(self, cache_key):
        """
        캐시에 값이 있을 때 redis에 접근하여 값을 반환함.
        :param cache_key: redis에서 불러올 값의 키 값.
        :return: 캐시에 저장된 데이터.
        """
        if type(cache_key) is not str:
            cache_key = str(cache_key)

        data = self.redis_conn.get(cache_key)

        # 디코딩해서 넘겨줌
        data = json.loads(data.decode('utf-8'))

        # 캐시 만료 시간 한번 더 업데이트
        self.update_expire_time(cache_key)

        return data


    def flush_old_data(self):
        """
        현재 DB에 있는 캐시 데이터를 전부 삭제함
        """
        # 현재 선택된 db 값을 모두 삭제함
        self.redis_conn.flushdb()
        return True

    def is_cached(self, key_value):
        """
        redis에 값이 있는지 확인한다.
        :param key_value: 확인할 key 값
        :return: 값이 있으면 True, 없으면 False 반환
        """

        if type(key_value) is not str:
            key_value = str(key_value)

        if self.redis_conn.exists(key_value):

            # hit 되고 나서 expire 되는 것을 방지하기 위한 update 호출출
            self.update_expire_time(key_value)
            print("cache hit")
            return True
        else:
            print("cache no hit")
            return False

    def update_expire_time(self, key_value):
        """
        redis에 해당 key에 해당하는 데이터의 expire time을 expire_time으로 초기화 한다.
        :param key_value: 업데이트 할 key 값
        """

        if type(key_value) is not str:
            key_value = str(key_value)

        self.redis_conn.expire(key_value, expire_time)

        return True



    def insert_data(self, key, data):
        """
        질의 결과를 캐시에 저장함.

        :param key: 사용자 입력 쿼리
        :param data: 검색 결과 data
        """
        if type(key) is not str:
            key = str(key)
        #print("keys: ",key)
        # 다른 곳에서 먼저 저장될 경우를 대비하여 다시 한번 확인
        if self.is_cached(key) is False:  # 값이 없는 경우

            # 저장을 위한 data 변환
            data = json.dumps(data, ensure_ascii=False).encode('utf-8')

            # 캐시 테이블에 값 추가
            self.redis_conn.setex(key,  expire_time, data)
            return True
        else:
            print("캐싱된 값이 이미 존재함.")
            return False

    def modify_data(self, data, query_id=1):
        """
        해당하는 id에 있는 dataset을 수정함. facet filter가 적용된 데이터셋 수정용이므로 1 인덱스의 값만 수정

        :param data: (list in dict) 변경할 데이터셋
        :param query_id:(int) 변경할 인덱스, default 1
        """
        self.cache_value_table[query_id] = data

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

        #print("keys: ", key)

        if self.is_cached(key):
            self.update_expire_time(key)
            data = self.redis_conn.get(key).decode('utf-8')
            data = json.loads(data)
        else:
            print("저장되지 않은 캐시입니다.")
            return []


        result = []
        if len(data) < end:
            end = len(data)
        for i in range(start - 1, end):
            result.append(data[i])
        return result, len(data)

    def get_all_data(self, cache_key):
        """
        facet class에게 facet key, value를 추출하기 위한 전체 데이터셋을 전달해주는 함수

        :param query_id: (int) 검색 결과를 불러올 query_id
        :return: (list in dict) query_id에 저장된 전체 데이터셋 리스트
        """
        # 잘못된 쿼리 아이디가 들어올 경우의 예외처리
        if self.redis_conn.exists(cache_key):
            self.update_expire_time(cache_key)
            data = self.redis_conn.get(cache_key).decode('utf-8')
            data = json.loads(data)
            self.update_expire_time(cache_key)
            return data
        else:
            return -1
