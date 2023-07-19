from flask import jsonify


def error_response(message, dev_error_message, status_code):
    """
    에러 메시지 반환 함수
    :Params        message: 사용자에게 보여줄 에러 메시지
                   dev_error_message: 개발자가 확인할 수 있는 에러 메시지
                   status_code: HTTP 상태 코드
    :Return        json: 에러 메시지를 포함하는 JSON 데이터를 반환
    """

    response = jsonify({
        "message": message
    })

    print(f"dev_error_message: {dev_error_message}")

    response.status_code = status_code
    return response