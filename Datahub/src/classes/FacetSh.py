from pprint import pprint
from .FacetExtractor import FacetExtractor, sentence_extractor

conf = {
    'facet_key': ['title', 'desc', 'date'],
    'string_key': ['title', 'desc'],
    'min_word_length': 2
}

class Facet:
    def __init__(self, datasets):
        """
        name: Facet 클래스 생성자
        desc: Metadataset들을 매개변수로 하여 Facet class를 생성
        """
        self.__original_datasets = list()
        self.set_original_datasets(datasets)

        self.__facet_key = self.make_facet_key()
        self.__facet_value = self.make_facet_value()
        self.__facet = self.make_facet()

    ''' Belows are Logic Methods '''

    def facet_query(self, text):
        facet = self.__facet
        value_keywords = facet["keywords"]
        query_keyword = sentence_extractor(text)
        keyindexlist = self.keyword_search(value_keywords, query_keyword)

        print("query:", text)
        print("facets:")
        for i in range(len(keyindexlist)):
            print(value_keywords[i])

    def keyword_search(self, value_keyword, query_keyword):
        keyindexlist = set()
        for qk in query_keyword:
            for i in range(len(value_keyword)):
                if qk in value_keyword[i]:
                    keyindexlist.add(i)

        return keyindexlist

    def extract_facet(self, input_text, facets):
        return

    ''' Belows are Maker Methods '''

    def make_facet_key(self):
        return conf['facet_key']

    def make_facet_value(self):
        facet_value = dict()

        for key in self.__facet_key:
            if key in conf['string_key']:
                key = 'string'
            facet_value[key] = list()

        facet_value["keywords"] = list()
        for dataset in self.__original_datasets:
            facet_value["keywords"].append(dataset["basicMetadata"]["keywords"])

        for dataset in self.__original_datasets:
            for key in self.__facet_key:
                if key in conf['string_key']:
                    facet_key = 'string'
                    dataset_key = key
                else:
                    facet_key = dataset_key = key
                t = dataset["basicMetadata"][dataset_key]
                facet_value[facet_key].append(dataset["basicMetadata"][dataset_key])

        return facet_value

    def make_facet(self):
        facet = dict()
        extractor = FacetExtractor()
        for key, value in self.__facet_value.items():
            if key == 'string':
                facet[key] = extractor.extract_string_tag(value)
            elif key == 'date':
                facet[key] = extractor.extract_date_facet(value)
            elif key == 'keywords':
                facet[key] = value

        return facet

    ''' Belows are Setter Methods '''

    def set_original_datasets(self, datasets):
        self.__original_datasets = datasets

    ''' Belows are Getter Methods '''

    def get_original_datasets(self):
        return self.__original_datasets

    def get_facet_key(self):
        return self.__facet_key

    def get_facet_value(self):
        return self.__facet_value

    def get_facet(self):
        return self.__facet

    ''' Belows are Print Methods '''

    def print_original_datasets(self):
        pprint(self.__original_datasets)

    def print_facet_key(self):
        pprint(self.__facet_key)

    def print_facet_value(self):
        pprint(self.__facet_value)

    def print_facet(self):
        pprint(self.__facet)
