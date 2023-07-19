flag = 1
sents = []



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
        word_list = phrase.split('_')
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
            result.append("_".join(phrase))
    #print(f"result:::: {result}")
    return result

# desc : 명사 및 명사구에서 stopwords에 해당되는 것들을 제거
# input : phrase_list : 명사 및 명사구 리스트
# output : description에 명시된 것과 같이 전처리된 결과 리스트
def delete_stopwords_in_phrase(phrase_list):
    result = []
    # 명사구 및 명사구의 단어 단위로 쪼개어 해당 단어가 stopword이면 제거
    for phrase in phrase_list:
        word_list = phrase.split('_')
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
            result.append("_".join(phrase))
    return result

# desc : okt 형태소 분석기를 통해 추출된 명사구 및 명사 리스트를 입력받아 숫자, stopword 제거
# input : phrase_list : okt 형태소 분석기를 추출된 명사구 및 명사 리스트
# output : stopword와 숫자가 제거된 리스트
def preprocess_phrase_version(phrase_list):
    #print("ori:", phrase_list)
    temp = delete_stopwords_in_phrase(phrase_list)
    #print("tmp:", temp)
    result = delete_number_in_phrase(temp)
    #print("re:", result)
    return result

# desc : okt 형태소 분석기를 통해 추출된 명사 리스트를 입력받아 그 중 stopwords에 해당되는 것이 있으면 제거하는 메소드
# input : noun_list : okt 형태 분석기를 통해 추출된 명사 리스트
# output : stopword가 제거된 리스트
def preprocess_noun_version(noun_list):
    #print("noun:", noun_list)
    result = delete_stopwords(noun_list)
    return result


with open("combine_processing_delete_new_line.txt", 'r', encoding='utf-8') as fin:
    with open("combine_processing_stopword.txt", 'w', encoding='utf-8') as fout:
        while 1:
            line = fin.readline()
            if line == '\n':
                flag += 1
                continue
            if flag%10==0:
                print(flag)
            if not line: break
            if '_' in line:
                line = line.split()
                phrases = preprocess_phrase_version(line)
                #sents.append(words)
                fout.write(" ".join(phrases)+'\n')
            else:
                words = line.split()
                nouns = preprocess_noun_version(words)
                #sents.append(words)
                fout.write(" ".join(nouns)+'\n')
            flag += 1