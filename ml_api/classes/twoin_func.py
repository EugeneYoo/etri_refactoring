ALLOWED_EXTENSIONS = {'txt', 'csv'}

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_train_data(flag, data_no):
    """
    사용할 데이터의 정보를 입력받아 해당 데이터의 경로를 반환
    :param flag: "w2v", "qds"로 두 가지 데이터를 선택 가능
    :param data_no: 사용할 데이터의 정보
    :return:
    """
    if flag =="w2v":
        if data_no >0 and data_no <= 3:
            return "w2v_train_data"+ str(data_no)+".txt"
    if flag =="qds":
        if data_no > 0 and data_no < 3:
            return "qds_train_data" + str(data_no)+".csv"

def calc_metric(data, total_len,gt):
    correct_count = 0
    data_len = len(data)

    if data_len is 0 or total_len is 0:
        #recall, precision
        return 0, 0

    # p@10 확인을 위한 데이터 slice
    if data_len >= 10:
        top10 = data[:10]
    else:
        top10 = data

    # gt와 일치하는 데이터를 확인하는 구간
    for dset in top10:
        if dset['uid'] in gt:
            correct_count+=1
    # 일치하는 값이 없는 경우에 대한 예외 처리
    if correct_count is 0:
        # recall, precision
        return 0, 0

    # p@k에서 data_len이 10 이상인 경우 10 이하인 경우 해당 데이터의 길이로 맞춤
    precision_base = 10 if data_len >= 10 else data_len

    precision = float(correct_count / precision_base)
    recall = float(correct_count / total_len)

    return recall, precision