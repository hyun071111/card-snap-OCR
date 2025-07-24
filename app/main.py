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
    """이미지 바이트를 받아 카드 번호와 유효 기간을 인식하는 함수"""
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

        card_number = "인식 실패"
        expiry_date = "인식 실패"

        # EasyOCR로 텍스트와 위치 정보(바운딩 박스)를 함께 추출
        results = reader.readtext(gray, detail=1)

        # 1. 카드 번호 위치 탐색 및 추출
        card_number_box = None
        for bbox, text, prob in results:
            cleaned_text = re.sub(r"\D", "", text)
            if len(cleaned_text) >= 15 and is_luhn_valid(cleaned_text):
                # Luhn 알고리즘을 통과하는 15자리 이상의 번호를 카드 번호로 확정
                card_number = cleaned_text
                card_number_box = bbox
                break  # 가장 먼저 찾은 유효한 번호를 사용

        # 2. 유효기간 지능형 탐색 (카드 번호를 찾았을 경우)
        if card_number_box:
            # 카드 번호 영역 바로 아래를 ROI(관심 영역)로 설정
            top_left = card_number_box[0]
            bottom_right = card_number_box[2]

            # Y좌표는 카드번호 바로 아래부터, X좌표는 카드번호와 비슷하게 설정
            roi_y_start = int(bottom_right[1])
            roi_y_end = int(
                bottom_right[1] + (bottom_right[1] - top_left[1]) * 2
            )  # ROI 높이는 카드번호 높이의 2배
            roi_x_start = int(top_left[0] - 20)  # 약간의 여유 공간
            roi_x_end = int(bottom_right[0] + 20)

            date_roi = gray[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            # 작게 잘라낸 유효기간 영역에 대해서만 Tesseract OCR 수행
            if date_roi.size > 0:
                conf_tess_date = r"--oem 3 --psm 7"  # 단일 텍스트 라인으로 취급
                date_text = pytesseract.image_to_string(date_roi, config=conf_tess_date)
                expiry_match = re.search(
                    r"\b(0[1-9]|1[0-2])\s?\/?\s?(\d{2})\b", date_text
                )
                if expiry_match:
                    expiry_date = f"{expiry_match.group(1)}/{expiry_match.group(2)}"

        # 3. Fallback: 만약 위 방법으로 정보를 못 찾았다면, 전체 텍스트에서 다시 검색
        if card_number == "인식 실패" or expiry_date == "인식 실패":
            combined_text = "\n".join(
                [res[1] for res in results]
            )  # EasyOCR 결과만으로 재구성

            if card_number == "인식 실패":
                digits_only = re.sub(r"\D", "", combined_text)
                matches = re.findall(r"(\d{15,16})", digits_only)
                if matches:
                    valid_numbers = [num for num in matches if is_luhn_valid(num)]
                    if valid_numbers:
                        card_number = max(valid_numbers, key=len)
                    else:
                        card_number = max(matches, key=len)

            if expiry_date == "인식 실패":
                fallback_match = re.search(
                    r"\b(0[1-9]|1[0-2])\s?\/?\s?(\d{2})\b", combined_text
                )
                if fallback_match:
                    expiry_date = f"{fallback_match.group(1)}/{fallback_match.group(2)}"

        return {
            "card_number": card_number,
            "expiry_date": expiry_date,
            "raw_text_combined": "\n".join([res[1] for res in results]).strip(),
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
