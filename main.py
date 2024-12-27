from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from openai import OpenAI

import openai
import os
import base64, json, re, time

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
client = openai.OpenAI()

prompt_template = """
당신은 사용자가 왜 이미지를 저장했는지 의도를 파악하는 조수입니다. 
첨부된 사진의 content를 분석하여, 사용자가 이미지를 저장한 이유를 classification 하세요. 
class의 종류는 '쇼핑, 문서, 음악, 기타'으로 총 4개가 있습니다. 
다음 예시를 참고하여 4개의 class 중 하나를 고른 후, 그 결과를 json 포맷으로 반환하세요.{{"className":"className"}} :
--------------
예시:
사진 속에 옷, 화장품 등의 상품 정보가 포함되어있다면 '쇼핑'으로 classify 합니다.
사진 속에 블로그 글, 책 글귀 등의 텍스트가 포함되어있다면 '문서'로 classify 합니다.
사진 속에 사용자가 듣던 음악에 정보가 포함되어있다면 '음악'으로 classify 합니다.
'쇼핑, 문서, 음악'의 class 중에서 선택하기 힘들다면 '기타'로 classify 합니다.
"""

# 응답 데이터 검증
def validate_response(response):
  
    valid_classes = {"쇼핑", "문서", "음악", "기타"}
    if not isinstance(response, dict):
        raise ValueError("응답 데이터가 JSON 형식이 아닙니다.")
    if "className" not in response:
        raise ValueError("응답 데이터에 'className' 키가 없습니다.")
    if response["className"] not in valid_classes:
        raise ValueError(f"유효하지 않은 className: {response['className']}. 허용 값: {valid_classes}")

# 실패시 재시도
async def analyze_description_with_retry(image_file: UploadFile, max_retries: int = 3, delay: float = 2.0):

    retries = 0
    while retries < max_retries:
        try:
            # OpenAI API 호출
            result = await analyze_description(image_file)

            # 데이터 검증
            validate_response(result)

            # 검증에 성공하면 결과 반환
            return result

        except (ValueError, HTTPException) as e:
            retries += 1
            if retries >= max_retries:
                raise HTTPException(
                    status_code=500,
                    detail=f"분석 실패: {str(e)} (재시도 횟수 초과)"
                )
            # 재시도 전 대기
            time.sleep(delay)

# GPT 이미지 분석석
async def analyze_description(image_file: UploadFile):
    try:
        
        image_data = await image_file.read()
        base64_image = encode_image(image_data)

        # OpenAI의 텍스트 분석 API 호출
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": prompt_template
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
        )

        result = response.choices[0].message.content.strip()
        # 응답 내용 추출
        response_json = extract_json_from_text(result)
        return response_json

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 실패: {str(e)}")

# 이미지 base64 인코딩딩
def encode_image(file_data: bytes) -> str:
  
    return base64.b64encode(file_data).decode("utf-8")

# JSON 데이터 추출 
def extract_json_from_text(text):
    """
    OpenAI 응답에서 JSON 데이터를 추출하여 반환합니다.
    """
    # ```json ... ``` 패턴 추출
    pattern = r'```json\s*(\{.*?\}|\[.*?\])\s*```'
    try:
        # 정규식으로 JSON 블록 추출
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)  # JSON 디코딩
        else:
            # ```json 패턴이 없을 경우 전체 텍스트를 파싱 시도
            return json.loads(text)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"JSON 디코딩 오류: {e}\n응답 내용: {text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"JSON 데이터 처리 오류: {e}\n응답 내용: {text}"
        )

@app.post("/api/tags")
async def upload_image(file: UploadFile = File(...)):
    try:
        # 사용자가 제공한 이미지 설명으로 분석
        result = await analyze_description_with_retry(file)

        # 분석 결과를 반환
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 업로드 실패: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Hello World!"}
