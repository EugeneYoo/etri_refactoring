import os
import logging
import psycopg2
import requests

from Datahub.src.classes.config_phrase import args
env = 'development' if not 'APP_ENV' in os.environ else os.environ['APP_ENV']
config = args[env]

logging.basicConfig(level=config['python_logging_lvl'])

class LogDB:
    def __init__(self):
        self.conn = psycopg2.connect(host=config['logging_server'], dbname=config['logging_server_dbname'], user=config['logging_server_user_id'],\
                                    password=['logging_server_password'], port=config['logging_server_port'])

    def parse(self, data):
        """
        
        pgsql 스키마 순서대로 맞춘 parser
        pgsql 스키마 순서가 변경되면 해당 순서에 맞게 변경해야 함

        """
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
                        from INFORMATION_SCHEMA.COLUMNS where table_name = 'logstore_test';
                    ''')
        data = curr.fetchall()
        logging.debug(data)

    def test(self):
        curr = self.conn.cursor()

        curr.execute(f'''
                        SELECT * FROM logstore_test where loggerclass = 'ClickDataLogger' order by datetime desc
                    ''')
        # COMMIT THE REQUESTS IN QUEUE

        data = curr.fetchall()

        logging.debug(data)

    def get_all_logdata(self):
        curr = self.conn.cursor()


        #sql = '''SELECT * FROM logstore_test where loggerclass  = 'ClickDataLogger' order by datetime desc limit 50'''
        sql = "SELECT A.message as Ames, B.message as Bmes, A.eventid as ID  FROM logstore_test as A join logstore_test as B on A.eventid = B.eventid where A.loggerclass  = 'ClickDataLogger' and B.loggerclass  = 'ListDataLogger' order by A.datetime desc limit 50"
        sql1 = "SELECT S.Ames as a, S.Bmes as b, S.ID as c, T.message as d " + "from (" + sql + ") as S join logstore_test as T on S.ID = T.eventid where T.loggerclass = 'KeyValueDataLogger' limit 50"

        logging.debug(sql1)

        curr.execute(sql1)
        # COMMIT THE REQUESTS IN QUEUE

        datas = curr.fetchall()

        li = []
        for data in datas:
            dic = {}
            dic['clickDatasetId'] = data[0][0]['clickDatasetId']
            dic['datasetList'] = data[1][0]['datasetList']
            dic['searchValue'] = data[3][0]['searchValue']
            
            li.append(dic)


        self.conn.close()

        return li
    def make_data(self, id_list):
        result = []
        for id in id_list:
            url = "http://14.37.211.69:7600/discovery/datasets/id/" + str(id)
            response = requests.get(url).text
            result.append(response)
        return result

    def get_train_data(self):
        log_list = self.get_all_logdata()
        logging.debug(log_list[0]['datasetList'])

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
                tmp_list.append(tmp[0])

            dataset_list.append(tmp_list)

        # click id
        # datasetlist -=> 검색으로 화면에 출력된 것
        # query string 

        return query_list, dataset_list, query




if __name__ == '__main__':
    DB = LogDB()


    click_id, datasetlist, query_string = DB.get_train_data()





