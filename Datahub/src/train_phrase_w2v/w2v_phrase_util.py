# coding: utf-8

import argparse
import codecs
import os
import lxml.etree as ET
import regex

from konlpy.tag import Mecab

analyzer = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")  # Mecab 형태소 분석기 로딩

# arguments setting
parser = argparse.ArgumentParser()
parser.add_argument('--lcode', help='ISO 639-1 code of target language. See `lcodes.txt`.')
parser.add_argument('--max_corpus_size', type=int, default=1000000000,
                    help='the maximum size of the corpus. Feel free to adjust it according to your computing power.')
args = parser.parse_args()

# lcode = args.lcode
lcode = 'ko'
if lcode == 'ko':
    from konlpy.tag import Okt  # pip install konlpy. See http://konlpy.org/en/v0.4.4/ for further information.

Okt = Okt()

max_corpus_size = args.max_corpus_size

# 한국어 stopwords를 담을 리스트
stopwords = []
file = open("../korean-stopwords.txt", 'r', encoding='utf-8')
# 한국어 stopwords 삽입
for word in file:
    stopwords.append(word)

# stopwords 중복 제거
stopwords = list(set(stopwords))
# stopwords에서 \n 등의 문자 제거
for i in range(len(stopwords)):
    stopwords[i] = stopwords[i].strip()


# desc : 단어 리스트를 입력받아 그 중에 stopwords에 해당되는 것이 있으면 삭제
# input : word_list : 학습 대상이 될 단어 리스트
def delete_stopwords(word_list):
    result = []
    for word in word_list:
        if word in stopwords:
            continue
        else:
            result.append(word)
    return result


# desc : 단위성 의존 명사와 함께 있는 숫자아닌 모든 숫자가 포함된 명사구의 경우 숫자를 제거
# input : phrase_list : 명사 및 명사구들이 포함된 리스트
# output : description에 명시된 것과 같이 전처리된 결과 리스트
def delete_number_in_phrase(phrase_list):
    result = []
    # 명사구 및 명사구의 단어 단위로 쪼개어 해당 단어가 숫자이면 제거
    for phrase in phrase_list:
        word_list = phrase.split()
        if len(word_list) == 1:
            if word_list[0].isdigit():
                continue
            else:
                result.append(word_list[0])
        elif len(word_list) == 0:
            continue
        else:
            phrase = []
            for word in word_list:
                if word.isdigit():
                    continue
                else:
                    phrase.append(word)
            result.append('_'.join(phrase))
    return result


# desc : 명사 및 명사구에서 stopwords에 해당되는 것들을 제거
# input : phrase_list : 명사 및 명사구 리스트
# output : description에 명시된 것과 같이 전처리된 결과 리스트
def delete_stopwords_in_phrase(phrase_list):
    result = []
    # 명사구 및 명사구의 단어 단위로 쪼개어 해당 단어가 stopword이면 제거
    for phrase in phrase_list:
        word_list = phrase.split()
        if len(word_list) == 1:
            if word_list[0] in stopwords:
                continue
            else:
                result.append(word_list[0])
        elif len(word_list) == 0:
            continue
        else:
            phrase = []
            for word in word_list:
                if word in stopwords:
                    continue
                else:
                    phrase.append(word)
            result.append('_'.join(phrase))
    return result


# desc : okt 형태소 분석기를 통해 추출된 명사구 및 명사 리스트를 입력받아 숫자, stopword 제거
# input : phrase_list : okt 형태소 분석기를 추출된 명사구 및 명사 리스트
# output : stopword와 숫자가 제거된 리스트
def preprocess_phrase_version(phrase_list):
    temp = delete_stopwords_in_phrase(phrase_list)
    result = delete_number_in_phrase(temp)
    return result


# desc : okt 형태소 분석기를 통해 추출된 명사 리스트를 입력받아 그 중 stopwords에 해당되는 것이 있으면 제거하는 메소드
# input : noun_list : okt 형태 분석기를 통해 추출된 명사 리스트
# output : stopword가 제거된 리스트
def preprocess_noun_version(noun_list):
    result = delete_stopwords(noun_list)
    return result


# desc : 코퍼스에서 데이터를 읽어서 학습 가능한 형태로 변환하는 메소드
# input : corpus_file_path : 코퍼스 파일 주소
# output : 최종 전처리된 학습 데이터 리스트
def preprocessTrainData(corpus_file_path):
    file = open(corpus_file_path, 'r', encoding='utf-8')
    # 학습 가능한 형태의 학습 데이터를 저장할 리스트
    result = []
    i = 0
    for line in file:
        # 한 라인씩 읽으면서 해당 라인의 명사와 명사구들을 추출하여 저장
        line = line.strip()
        phrases = preprocess_phrase_version(Okt.phrases(line))
        nouns = preprocess_noun_version(Okt.nouns(line))
        if len(phrases) > 0:
            result.append(phrases)
        if len(nouns) > 0:
            result.append(nouns)
        i += 1
    return result


def clean_text(text):
    global lcode

    # Common
    text = regex.sub("(?s)<ref>.+?</ref>", "", text)  # remove reference links
    text = regex.sub("(?s)<[^>]+>", "", text)  # remove html tags
    text = regex.sub("&[a-z]+;", "", text)  # remove html entities
    text = regex.sub("(?s){{.+?}}", "", text)  # remove markup tags
    text = regex.sub("(?s){.+?}", "", text)  # remove markup tags
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text)  # remove link target strings
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text)  # remove media links

    text = regex.sub("[']{5}", "", text)  # remove italic+bold symbols
    text = regex.sub("[']{3}", "", text)  # remove bold symbols
    text = regex.sub("[']{2}", "", text)  # remove italic symbols

    if lcode in ['ko']:  # korean
        text = regex.sub(u"[^ \r\n\p{Hangul}.?!]", " ", text)  # Replace unacceptable characters with a space.
    else:  # Mostly european languages
        text = regex.sub(u"[^ \r\n\p{Latin}\-'‘’.?!]", " ", text)
        text = text.lower()

    # Common
    text = regex.sub("[ ]{2,}", " ", text)  # Squeeze spaces.
    return text


def sentence_segment(text):
    '''
    Args:
      text: A string. A unsegmented paragraph.

    Returns:
      A list of sentences.
    '''
    sents = regex.split("([.?!])?[\n]+|[.?!] ", text)
    return sents


def word_segment(sent):
    '''
    Args:
      sent: A string. A sentence.

    Returns:
      A list of words.
    '''
    global lcode
    if lcode in ['ko']:
        words = [word for word in analyzer.nouns(sent)]
        result = []
        for word in words:
            if word in stopwords:
                continue
            else:
                result.append(word)
    else:  # Mostly european languages
        result = sent.split()

    return result


def phrase_segment(sent):
    '''
    Args:
      sent: A string. A sentence.

    Returns:
      A list of words.
    '''
    global lcode
    if lcode in ['ko']:
        words = [word for word in Okt.phrases(sent)]
        result = []
        for word in words:
            if word in stopwords:
                continue
            else:
                result.append(word.replace(' ', '_'))

    return result


def build_corpus():
    global lcode, max_corpus_size, fname
    with codecs.open("combine_processing_limit.txt", 'w', 'utf-8') as fout:
        i = 1
        with codecs.open("combine.txt", 'r', 'utf-8') as fin:
            while 1:
                line = fin.readline()
                if not line: break
                try:
                    running_text = clean_text(line)
                    sents = sentence_segment(running_text)
                    for sent in sents:
                        if sent is not None:
                            words = word_segment(sent)
                            fout.write(" ".join(words) + "\n")
                            p_words = phrase_segment(sent)
                            fout.write(" ".join(p_words) + "\n")
                except:
                    print("에러 컨티뉴")
                    continue  # it's okay as we have a pretty big corpus!
                if i % 1000 == 0:
                    print(i),
                    fsize = os.path.getsize("combine_processing_limit.txt")
                    if fsize > max_corpus_size:
                        break
                i += 1


def delete_new_line():
    with open("combine_processing_limit.txt", 'r', encoding='utf-8') as fin:
        with open("combine_processing_delete_new_line.txt", 'w', encoding='utf-8') as fout:
            while 1:
                line = fin.readline()
                if line == '\n':
                    continue
                if not line: break
                fout.write(line)

if __name__ == "__main__":
    #build_corpus()
    delete_new_line()
    print("Done")
