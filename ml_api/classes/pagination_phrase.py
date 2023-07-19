import json
import os
import redis
import threading
from .config_phrase import args


# Variables receiving environment settings
env = 'development' if not 'APP_ENV' in os.environ else os.environ['APP_ENV']
config=args[env]

# redis expire time
expire_time = config['redis_expire_time']


class Pagination:
    def __init__(self):
        """
        # Pagination class

        params
        ------
        redis_conn: redis connect to store cache

        query_count: (int) Acts as a dataset index stored in cache_value_table.\n

        """
        self.lock = threading.Lock()
        self.redis_conn = redis.StrictRedis(host=config['redis_server'], port=config['redis_port'], db=config['redis_db_number'])

    def flush_old_data(self):
        """
        Delete all cache data in the current DB
        """
        with self.lock:
            # Use Redis pipeline to execute flushdb command atomically
            pipeline = self.redis_conn.pipeline()
            pipeline.flushdb()
            pipeline.execute()

        return True

    def update_expire_time(self, key_value):
        """
        Initialize the expire time of the data corresponding to the key in redis to expire_time.
        :param key_value: key value to update
        """

        if type(key_value) is not str:
            key_value = str(key_value)

        with self.redis_conn.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key_value)
                    pipe.multi()
                    pipe.expire(key_value, expire_time)
                    pipe.execute()
                    break
                except redis.WatchError:
                    continue

        return True

    def insert_data(self, key, data):
        """
        Store query results in cache.

        :param key: query user input
        :param data: search result data
        """
        if type(key) is not str:
            key = str(key)
        #print("keys: ",key)
        # Double check in case it gets saved elsewhere first

        with self.redis_conn.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    if pipe.exists(key) is False:
                        # Convert data for saving
                        data = json.dumps(data, ensure_ascii=False).encode('utf-8')

                        # Add value to cache table
                        pipe.multi()
                        pipe.setex(key, expire_time, data)
                        pipe.execute()
                        return True
                    else:
                        print("Cached value already exists.")
                        return False
                except redis.WatchError:
                    continue

    def get_data(self, key, start, end):
        """
        It loads the cached dataset through the key and passes the paginated data.
        The total size of the data is also transmitted (total_page calculation).

        :param key: (int) Key to retrieve search results from
        :param start: (int) pagination start page
        :param end: (int) pagination end page
        :return: (list in dict) List of datasets with pagination applied,
                data_length: stored data length
        """

        if type(key) is not str:
            key = str(key)

        #print("keys: ", key)

        if self.is_cached(key):
            self.update_expire_time(key)
            data = self.redis_conn.get(key).decode('utf-8')
            data = json.loads(data)
        else:
            print("This is an unsaved cache.")
            return []
        result = []
        if len(data) < end:
            end = len(data)
        for i in range(start - 1, end):
            result. append(data[i])
        return result, len(data)


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

    def get_all_data(self, cache_key):
        """
        A thread-safe function that passes the entire dataset to extract facet keys and values to the facet class.

        :param cache_key: (str) cache key to retrieve search results from
        :return: (list of dict) list of all datasets stored in the cache_key, or -1 if cache_key does not exist
        """
        with self.redis_conn.pipeline() as pipe:
            while True:
                try:
                    # Watch the cache_key to detect changes
                    pipe.watch(cache_key)

                    # Check if the cache_key exists
                    if not pipe.exists(cache_key):
                        return -1

                    # Load the data from cache
                    data = pipe.get(cache_key)

                    # Begin a MULTI block to perform a transaction
                    pipe.multi()

                    # Update the expiration time of the cache_key
                    pipe.expire(cache_key, expire_time)

                    # Execute the transaction
                    pipe.execute()

                    # Convert the data to a list of dictionaries
                    data = json.loads(data.decode('utf-8'))

                    return data

                except redis.exceptions.WatchError:
                    # If the cache_key was modified by another client, retry
                    continue

                finally:
                    # Reset the pipeline
                    pipe.reset()