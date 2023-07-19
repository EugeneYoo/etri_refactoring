import os
import psycopg2
import requests
import logging

from .config_phrase import args
env = 'development' if not 'APP_ENV' in os.environ else os.environ['APP_ENV']
config = args[env]

logging.basicConfig(level=config['python_logging_lvl'])

class LogDB:
    def __init__(self):
        self.conn = psycopg2.connect(host=config['logging_server'], dbname=config['logging_server_dbname'], user=config['logging_server_user_id'],\
                                    password=['logging_server_password'], port=config['logging_server_port'])

    def parse(self, data):
        dic = {}
        dic['datetime'] = data[0]
        dic['loggertype'] = data[1]
        dic['loggerclass'] = data[2]
        dic['loggername'] = data[3]
        dic['logzone'] = data[4]
        dic['eventid'] = data[5]
        dic['userid'] = data[6]
        dic['message'] = data[7]

        return dic

    def check_table(self):
        curr = self.conn.cursor()

        curr.execute(f'''
                        select column_name, data_type, character_maximum_length, column_default, is_nullable
                        from INFORMATION_SCHEMA.COLUMNS where table_name = 'logstore_test2';
                    ''')
        datas = curr.fetchall()
        print(datas)

    def test(self):
        curr = self.conn.cursor()

        curr.execute(f'''
                        SELECT * FROM logstore_test2 where loggerclass = 'ClickDataLogger' order by datetime desc
                    ''')
        # COMMIT THE REQUESTS IN QUEUE

        datas = curr.fetchall()

        print(datas)

    def get_all_logdata(self):
        curr = self.conn.cursor()


        #sql = '''SELECT * FROM logstore_test2 where loggerclass  = 'ClickDataLogger' order by datetime desc limit 50'''
        sql = "SELECT A.message as Ames, B.message as Bmes, A.eventid as ID  FROM logstore_test2 as A join logstore_test2 as B on A.eventid = B.eventid where A.loggerclass  = 'ClickDataLogger' and B.loggerclass  = 'ListDataLogger' order by A.datetime desc limit 50"
        sql1 = "SELECT S.Ames as a, S.Bmes as b, S.ID as c, T.message as d " + "from (" + sql + ") as S join logstore_test2 as T on S.ID = T.eventid where T.loggerclass = 'KeyValueDataLogger' limit 50"

        print(sql1)

        curr.execute(sql1)
        # COMMIT THE REQUESTS IN QUEUE

        datas = curr.fetchall()

        li = []
        for data in datas:
            dic = {}
            dic['clickDatasetId'] = data[0][0]['clickDatasetId']
            dic['datasetList'] = data[1][0]['datasetList']
            dic['searchValue'] = data[3][0]['searchValue']
            #print(data[2])
            li.append(dic)


        self.conn.close()

        return li
    def make_data(self, id_list):
        result = []
        print(id_list)
        for id in id_list:
            url = "http://localhost:8600/discovery/datasets/id/" + str(id)
            #print(url)
            response = requests.get(url).text
            result.append(response)
        return result

    def get_train_data(self):
        # 원활한 테스트를 위해 잠금
        log_list = self.get_all_logdata()

        #log_list =
        query_list = []
        dataset_list = []
        query = []

        for lt in log_list:

            datasets = self.make_data(lt['datasetList'])

            query_list.append(lt['clickDatasetId'])
            query.append(lt['searchValue'])

            tmp_list = []
            for dataset in range(len(datasets)):
                tmp = eval(datasets[dataset])['results']
                tmp_list.append(tmp)

            dataset_list.append(tmp_list)

        return query_list, dataset_list, query




if __name__ == '__main__':
    DB = LogDB()
    # a: click id
    # b: datasetlist -=> 검색으로 화면에 출력된 것
    # C: query string

    a, b, c = DB.get_train_data()





