import os
import time
import cv2
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch
from models import load_model  # 모델 로드 함수 가져오기
import numpy as np

# 이미지 업스케일링
def upscale_image(input_path, output_path, model):
    if not os.path.exists(input_path):
        print(f"Error: Input image not found at {input_path}")
        return

    # 이미지 읽기
    input_image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if input_image is None:
        print(f"Error: Could not read image file {input_path}")
        return

    # OpenCV 이미지를 PIL 이미지로 변환
    input_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    input_tensor = ToTensor()(input_image).unsqueeze(0)

    # 시작 시간 기록
    start_time = time.time()

    print("Processing image...")
    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_image = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
    output_image = (output_image * 255).astype(np.uint8)

    # 저장
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_image)

    # 처리 완료 후 시간 계산
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Upscaled image saved to {output_path}")
    print(f"Processing time: {processing_time:.2f} seconds.")

# 실행
if __name__ == "__main__":
    model_path = "models/RRDB_ESRGAN_x4.pth"  # 모델 경로 설정
    input_image = "C:/work_lyj/upscale_video_proj/src/input_image.jpg"
    output_image = "C:/work_lyj/upscale_video_proj/src/output_image.jpg"

    print("모델 로드 중...")
    model = load_model(model_path)

    print("업스케일링 시작...")
    upscale_image(input_image, output_image, model)
    print("완료되었습니다!")