from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pytesseract
import re
import uvicorn
import easyocr

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
except Exception:
    print(
        "Tesseract not found. Please set the correct path if not on Windows or if installed elsewhere."
    )

try:
    reader = easyocr.Reader(["en", "ko"], gpu=False)
except Exception as e:
    print(f"EasyOCR 초기화 실패: {e}")
    reader = None


def rotate_image(image: np.ndarray) -> np.ndarray:
    """이미지 방향을 자동으로 감지하고 보정"""
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        rotation = osd["rotate"]
        if rotation != 0:
            if rotation == 90:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                return cv2.rotate(image, cv2.ROTATE_180)
            elif rotation == 270:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
    except Exception as e:
        print(f"Could not detect rotation, using original image. Error: {e}")
        return image


def ocr_card_image(image_bytes: bytes) -> dict:
    """Two-Pass OCR: 카드 번호를 먼저 찾고, 나머지 텍스트에서 유효기간을 탐색"""
    if reader is None:
        return {"error": "EasyOCR is not initialized."}

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Failed to decode image."}

        img = rotate_image(img)

        h, w, _ = img.shape
        img_resized = cv2.resize(
            img, (1000, int(h * 1000 / w)), interpolation=cv2.INTER_LANCZOS4
        )
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # EasyOCR로 텍스트와 위치 정보 모두 추출
        results = reader.readtext(gray, detail=1)

        card_number = "인식 실패"
        expiry_date = "인식 실패"
        card_number_index = -1

        # 1단계: 카드 번호 탐색 및 분리
        for i, (bbox, text, prob) in enumerate(results):
            cleaned_text = re.sub(r"\D", "", text)
            if len(cleaned_text) >= 15 and is_luhn_valid(cleaned_text):
                card_number = cleaned_text
                card_number_index = i
                break

        # 2단계: 유효기간 탐색 (카드 번호를 제외한 나머지 텍스트에서)
        for i, (bbox, text, prob) in enumerate(results):
            if i == card_number_index:
                continue  # 카드 번호 텍스트는 건너뜀

            expiry_match = re.search(r"\b(0[1-9]|1[0-2])\s?\/?\s?(\d{2})\b", text)
            if expiry_match:
                # MM/YY 패턴이 발견되면 바로 사용
                expiry_date = f"{expiry_match.group(1)}/{expiry_match.group(2)}"
                break

        # 3. Fallback: 만약 Tesseract가 더 잘 읽을 경우를 대비한 최종 안전장치
        if expiry_date == "인식 실패":
            conf_tess = r"--oem 3 --psm 6"
            tess_text = pytesseract.image_to_string(gray, config=conf_tess)
            fallback_match = re.search(
                r"\b(0[1-9]|1[0-2])\s?\/?\s?(\d{2})\b", tess_text
            )
            if fallback_match:
                expiry_date = f"{fallback_match.group(1)}/{fallback_match.group(2)}"

        return {
            "card_number": card_number,
            "expiry_date": expiry_date,
            "raw_text_easyocr": [res[1] for res in results],
        }

    except Exception as e:
        return {
            "error": "An unexpected error occurred during OCR processing.",
            "detail": str(e),
        }


def is_luhn_valid(card_number: str) -> bool:
    """Luhn 알고리즘(Modulus 10)으로 카드 번호 유효성 검사"""
    try:
        num_digits, s = len(card_number), 0
        for i, digit_char in enumerate(card_number):
            digit = int(digit_char)
            if (i % 2) == (num_digits % 2):
                digit *= 2
            if digit > 9:
                digit -= 9
            s += digit
        return (s % 10) == 0
    except (ValueError, TypeError):
        return False


@app.post("/api/ocr", tags=["Image OCR"])
async def extract_card_info_from_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = ocr_card_image(image_bytes)
    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
