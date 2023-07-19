class DataHubInfo:
    def __init__(self):
        """통합 검색 포털 metadata struct

        params
        -------------------------
        hub_name: (string) 데이터 허브 이름
        hub_url: (string)데이터 허브 url
        hub_categories: (list) 데이터 허브 도메인
        is_semantic: (bool) 데이터 허브의 시맨틱 검색 지원 여부
        """
        self.hub_name = ""
        self.hub_url = ""
        self.hub_categories = []
        self.is_semantic = False

    def __repr__(self):
        return '({}, {}, {}, {})'.format(self.hub_name, self.hub_url, self.hub_categories, self.is_semantic)

    def __eq__(self, other):
        if self.hub_name == other.hub_name:
            if self.hub_url == other.hub_url:
                return True
        else:
            return False
        return False

    def get_is_semantic(self):
        return self.is_semantic

    def get_hub_name(self):
        return self.hub_name

    def get_hub_url(self):
        return self.hub_url

    def get_hub_categories(self):
        return self.hub_categories

    def set_hub_categories(self, categories):
        # categories 저장 방식에 따라 변경.
        print(categories.values())
        temp = list(categories.values())
        self.hub_categories = temp

    def set_hub_url(self, url):
        self.hub_url = url

    def set_hub_name(self, name):
        self.hub_name = name

    def set_is_semantic(self, flag):
        self.is_semantic = bool(flag)