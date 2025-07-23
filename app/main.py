from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pytesseract
import re
import uvicorn
import traceback

# FastAPI 앱 생성
app = FastAPI()

# CORS 미들웨어 추가 (브라우저에서 API를 호출할 수 있도록 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 중요 ---
# 사용자 환경에 맞게 Tesseract OCR 실행 파일 경로를 설정해야 합니다.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def ocr_card_image(image_bytes: bytes) -> dict:
    """
    이미지에서 카드 번호와 유효기간을 추출합니다.
    관심 영역(ROI)을 찾아 인식률을 향상시키는 로직이 포함되어 있습니다.
    """
    try:
        # 1. 이미지 로드 및 기본 전처리
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        h, w = img.shape[:2]
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러로 노이즈 감소
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 외곽선 찾기
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        card_number_roi = None
        
        # 3. 찾은 외곽선 중 카드 번호 영역으로 추정되는 부분을 필터링
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            
            # 숫자 그룹의 특징(가로로 김, 적절한 크기)을 기반으로 필터링
            if aspect_ratio > 3.0 and 100 < w < 500 and 20 < h < 80:
                # ROI(관심 영역)를 원본 gray 이미지에서 잘라냄
                card_number_roi = gray[y:y+h, x:x+w]
                break
        
        text = ""
        if card_number_roi is not None:
            _, roi_thresh = cv2.threshold(card_number_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(roi_thresh, config=config)
        else:
            config = r'--psm 6 -c tessedit_char_whitelist=0123456789/'
            text = pytesseract.image_to_string(gray, config=config)

        # 5. 정규 표현식으로 정보 추출
        # OCR 결과에서 공백 등 숫자가 아닌 문자를 모두 제거
        digits_only = re.sub(r'\D', '', text)
        card_number = digits_only[:16] if len(digits_only) >= 16 else digits_only
        
        # 유효기간은 ROI가 아닌 전체 이미지에서 찾는 것이 더 안정적일 수 있음
        full_text_for_expiry = pytesseract.image_to_string(gray, config=r'--psm 6 -c tessedit_char_whitelist=0123456789/')
        expiry_match = re.search(r'\b(0[1-9]|1[0-2])\s?/?\s?(\d{2})\b', full_text_for_expiry)
        expiry_date = f"{expiry_match.group(1)}/{expiry_match.group(2)}" if expiry_match else None

        return {
            "card_number": card_number if card_number else "인식 실패",
            "expiry_date": expiry_date,
            "raw_text": text.strip()
        }

    except Exception as e:
        # 서버에서 오류 발생 시, 원인 파악을 위해 터미널에 로그 출력
        print("OCR 함수에서 오류 발생 !!!!!!!!!!!!")
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/api/ocr")
async def extract_card_info(file: UploadFile = File(...)):
    image_data = await file.read()
    result = ocr_card_image(image_data)
    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)