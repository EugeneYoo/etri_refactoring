from konlpy.tag import Twitter


class FacetExtractor:
    def __init__(self):
        self.okt = Twitter()

    def extract_string_tag(self, string_datas):
        extracted_string_tags = list()
        for string_data in string_datas:
            extracted_string_tags += self.okt.nouns(string_data)

        return list(set(extracted_string_tags))

    def extract_date_facet(self, date_datas):
        extracted_date_facet = {
            'year': [],
            'month': [],
            'day': []
        }
        for date in date_datas:
            extracted_date_facet['year'].append(int(date[0:4]))
            extracted_date_facet['month'].append(int(date[5:7]))
            extracted_date_facet['day'].append(int(date[8:10]))

        for key in extracted_date_facet:
            extracted_date_facet[key] = list(set(extracted_date_facet[key]))

        return extracted_date_facet


def sentence_extractor(text):
    okt = Okt()
    return okt.nouns(text)
