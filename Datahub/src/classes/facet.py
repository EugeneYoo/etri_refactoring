import os
import logging
from .config_phrase import args
from itertools import product

env = 'development' if not 'APP_ENV' in os.environ else os.environ['APP_ENV']
config = args[env]

logging.basicConfig(level=config['python_logging_lvl'])

class Facet:
    def __init__(self, datasets):
        """
        facet filter 클래스

        params
        ------

        original_dataset:(list) query_id를 통해 받아온 전체 검색 결과 데이터셋\n
        filtered_dataset:(list) original_dataset에서 facet filter가 적용된 데이터셋\n
        facet_key: (list)추출한 facet key(=category) ['type','publisher','tag']와 같이 facet 키 값을 저장함\n
        facet_value: (dict(string, list)) 추출한 facet value {'type':['a','b'],'publisher':['c','d'], 'tag':['e','f']}와 같이 facet 값을 저장함\n
        :param datasets: original_dataset을 초기화 하기 위한 datasets
        """

        self.original_datasets = []
        self.filtered_datasets = []
        # ['type','publisher','tag']와 같이 facet 키 값을 저장함
        self.facet_key = []
        # {'type':['a','b'],'publisher':['c','d'], 'tag':['e','f']}와 같이 facet 값을 저장함
        self.facet_value = {}

        # 오리지널 데이터를 pagination 객체에서 긁어옴
        self.init_original_datasets(datasets)
        # 데이터셋 로드, 키 리스트 만듦, facet 추출, 필터 결과, 페이지네이션 적용.
        self.make_facet_key()
        self.make_facet_value()
        self.apply_facet_filter()

    def init_original_datasets(self, datasets):
        """datasets으로 original_datasets를 초기화 함.

        :param datasets: original_dataset을 초기화할 datasets
        """
        self.original_datasets = datasets

    def make_facet_key(self):  # 수정
        """데이터셋에서 facet_key 추출 해서 self.facet_key에 저장\n

        ['type','publisher','tag']와 같이 facet 키 값을 저장함
        """
        key_list = []

        '''
        for dataset in self.original_datasets:
            print(dataset)
            for metadata_type in ['basicMetadata','extendedMetadata']:
                for metadata in dataset[metadata_type]:
                    # 속성 값의 뒷 부분만 떼서 facet key value로 사용
                    key = metadata['propertyId']

                    # .split(':')[1] -> split 은 나중에 출력시?
                    if key not in key_list:
                        key_list.append(key)
        key_list.remove('dcterms:title')
        key_list.remove('dcterms:description')
        '''
        for dataset in self.original_datasets:
            tmp_key_list = dataset.keys()
            for key in tmp_key_list:
                if key not in key_list:
                    key_list.append(key)

        key_list.remove('title')
        key_list.remove('desc')
        key_list.remove('accessURL')
        key_list.remove('uid')
        key_list.remove('@timestamp')
        key_list.remove('_score')
        try:
            key_list.remove('documentVector')
        except:
            pass
        logging.debug(key_list)

        self.facet_key = key_list

    def make_facet_product(self, dicts):
        """
        facet filter에 해당하는 경우의 수 생성\n

        :param dicts: facet value 값
        :return: facet filter 경우의 수 생성 결과가 저장된 제너레이터
        """
        return (dict(zip(dicts, x)) for x in product(*dicts.values()))

    def is_facet_value_in_data(self, facets, facet_conditions, data):
        """
        선택한 facet이 데이터셋에 들어있는가 확인

        :param facets: facet 전체
        :param facet_conditions: make_facet_product로 만들어진 facet 경우의 수
        :param data: 비교할 데이터셋
        :return: if data in facet_conditions == True, else False
        """

        key_list = facets.keys()
        facet_dic_for_compare = {}
        len_key = len(facets.keys())

        '''
        # 데이터셋의 facet 값을 추출
        for metadata_type in ['basicMetadata','extendedMetadata']:
            for metadata in data[metadata_type]:
                # value안의 type, id, value 값을 추출해서 비교해야 함.
                name = metadata['propertyId']
                value_list = []
                # value 안에 리스트 형식인 경우의 추출을 위함.
                for meta_value in metadata['value']:
                    print(name ,meta_value.keys(), metadata['value'])
                    if '@id' in meta_value.keys():
                        value_list.append(meta_value['@id'])
                    if '@type' in meta_value.keys():
                        value_list.append(meta_value['@type'])
                    if '@value' in meta_value.keys():
                        value_list.append(meta_value['@value'])

                # 기존에 이미 해당 key 값이 있는 경우 extend
                if name in facet_dic_for_compare.keys():
                    facet_dic_for_compare[name].extend(value_list)
                else:
                    # 없는 경우 바로 value 대입
                    facet_dic_for_compare[name] = value_list

        # facet이 일치하는 횟수를 셈.
        for value in facet_conditions:
            facet_match_cnt = 0
            for key in key_list:
                try:
                    # facet이 데이터 값에 있는 경우
                        if value[key] in facet_dic_for_compare[key]:
                            print(value[key])
                            print(facet_dic_for_compare[key])
                            facet_match_cnt += 1
                except: # 해당하는 facet이 없는 경우 오류가 발생하지 않게함.
                    pass
            # 조건이 전부 일치하면 True
            if facet_match_cnt == len_key:
                return True
        return False        
        '''
        for metadata_type in ['modified_date', 'publisher', 'tag']:
                facet_dic_for_compare[metadata_type] = data[metadata_type]

        # facet이 일치하는 횟수를 셈.
        for value in facet_conditions:
            facet_match_cnt = 0
            for key in key_list:
                try:
                    # facet이 데이터 값에 있는 경우
                    if value[key] in facet_dic_for_compare[key]:
                        logging.debug(value[key])
                        logging.debug(facet_dic_for_compare[key])
                        facet_match_cnt += 1
                except:  # 해당하는 facet이 없는 경우 오류가 발생하지 않게함.
                    pass
            # 조건이 전부 일치하면 True
            if facet_match_cnt == len_key:
                return True
        return False

    def make_facet_value(self):
        """
        전체 데이터셋에서 facet value를 추출함.
        """

        if self.original_datasets is "":
            logging.debug("facet을 추출할 데이터셋이 없음")
            return print("facet을 추출할 데이터셋이 없음")

        facet_dic = dict.fromkeys(self.facet_key, [])
        for key in self.facet_key:
            value_list = []
            for data in self.original_datasets:
                facet_values = data[key]
                facet_value = facet_values.split()
                for value in facet_value:
                    if value not in value_list:
                        if len(value) > 1:
                            value_list.append(value)
            facet_dic[key] = value_list
        self.facet_value = facet_dic
        '''
        facet_list = [] # Metadata 전체 값을 임시로 저장하는 리스트
        facet_dic = {} # 중복을 제거한 facet_value dictionary


        # basicMetadata의 데이터만 추출
        for data in self.original_datasets:
            if data.get('basicMetadata') is not None:
                facet_list.extend(data['basicMetadata'])

        # extendedMetadata 데이터만 추출
        for data in self.original_datasets:
            if data.get('extendedMetadata') is not None:
                facet_list.extend(data['extendedMetadata'])

        for data in facet_list:
            name = data['propertyId']
            value_list = []
            for value in data['value']:
                if '@id' in value.keys():
                    value_list.append(value['@id'])
                if '@type' in value.keys():
                    value_list.append(value['@type'])
                if '@value' in value.keys():
                    value_list.append(value['@value'])

            # propertyId가 facet_dic에 없는 경우
            if name not in facet_dic:
                deduplicate_value_list = []
                for value in value_list:
                    if value not in deduplicate_value_list:
                        deduplicate_value_list.append(value)
                facet_dic[name] = deduplicate_value_list
            else:
                for value in value_list:
                    if value not in facet_dic[name]:
                        facet_dic[name].append(value)

        facet_dic.pop('dcterms:title')
        facet_dic.pop('dcterms:description')

        self.facet_value = facet_dic
        '''

    def apply_facet_filter(self, select_facet_list=""):
        """
        선택한 facet에 해당되는 데이터셋만 걸러서 filtered_datasets에 저장함

        파싱 단계에선 input -> output 형태로 변환하여 내부에서 사용
        input =
        [
          {“facetName”: "fileType", “facetValues” ["csv”, "zip"]},
          {“facetName”: "publisher", “facetValues” ["경기도청", "보건복지부”]
        ]
        output =
        {
         {“fileType”: ["csv”, "zip"]},
         {"publisher": ["경기도청", "보건복지부”]}
         }


        :param select_facet_list: 선택한 facet values
        """

        # input을 파싱하는 단계
        result_dic = {}

        for facet in select_facet_list:
            result_dic[facet['facetName']] = facet['facetValues']
        # 파싱 종료

        # 실제 값이 들어온 facet들을 담는 변수
        facets = {}
        for key in result_dic:
            if len(result_dic[key]) is not 0:
                facets[key] = result_dic[key]

        # 선택된 facet이 없는 경우 filtered_datasets을 original_datasets로 초기화함.
        if len(facets) is 0:
            self.filtered_datasets = self.original_datasets
            return print("선택한 facet이 없습니다.")

        # facet filter에 해당되는 데이터셋을 담는 리스트
        result = []

        # 검색 결과 데이터셋이 없는 경우
        if len(self.original_datasets) == 0:
            return print("검색 결과 데이터셋이 없음")

        # facet의 경우의 수를 담는 리스트
        facet_condition = []

        # facet의 경우의 수를 담음.
        for i in self.make_facet_product(facets):
            facet_condition.append(i)

        logging.debug(facet_condition)

        # 오리지널 데이터셋에서 facet 필터링 진행
        for dataset in self.original_datasets:
            if self.is_facet_value_in_data(facets, facet_condition, dataset):
                # facet filter에 일치한 데이터셋이 결과 리스트에 없는 경우
                if dataset not in result:
                    # result에 추가함.
                    result.append(dataset)
        logging.debug(result)
        self.filtered_datasets = result

    def print_facet_key(self):
        """
        facet_key를 출력하는 함수
        """
        print(self.facet_key)

    def print_facet_value(self):
        """
        facet_value를 출력하는 함수
        """
        for key in self.facet_key:
            print("key: ", key, "\n", self.facet_value[key])

    def get_original_datasets(self):
        """
        original_datasets 전체를 불러옴
        :return: self.original_datasets
        """
        return self.original_datasets

    def get_filtered_datasets(self):
        """
        filtered_datasets 전체를 불러옴
        :return: self.filtered_datasets
        """
        return self.filtered_datasets

    def get_facet_value(self):
        """
        facet_value 전체를 불러옴
        :return: self.get_facet_value
        """
        return self.facet_value

    def facet_wrapper_for_response(self):
        """
        facet_value를 response에 맞는 형태로 감싸서 반환함
        :return:
        """
        return_list = []
        if len(self.facet_value) is not 0:
            facets = self.get_facet_value()
            for key in facets.keys():
                temp = {}
                temp['facetName'] = key
                temp['facetValues'] = facets[key]
                return_list.append(temp)
        else:
            return print("facet이 없습니다.")

        return return_list