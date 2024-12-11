import cv2
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import time
from tqdm import tqdm  # 진행률 표시를 위한 라이브러리
from models import load_model

# 프레임 업스케일링
def upscale_frame(frame, model):
    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = ToTensor()(input_image).unsqueeze(0)

    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_image = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
    return (output_image * 255).astype(np.uint8)

# 동영상 업스케일링
def upscale_video(input_path, output_path, model, upscale_size, max_frames=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return

    # FPS 및 총 프레임 수 확인
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Total Frames: {total_frames}")

    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
        print(f"Processing limited to {max_frames} frames.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, upscale_size)

    # 진행률 표시 및 시간 측정
    start_time = time.time()
    for frame_idx in tqdm(range(total_frames), desc="Upscaling video"):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {frame_idx + 1}")
            break

        upscaled_frame = upscale_frame(frame, model)
        out.write(cv2.cvtColor(upscaled_frame, cv2.COLOR_RGB2BGR))

        # 남은 시간 계산
        elapsed_time = time.time() - start_time
        avg_time_per_frame = elapsed_time / (frame_idx + 1)
        remaining_time = avg_time_per_frame * (total_frames - frame_idx - 1)
        tqdm.write(f"Processed frame {frame_idx + 1}/{total_frames}, Estimated time left: {remaining_time:.2f}s")

        # 프레임 제한 체크
        if max_frames is not None and frame_idx + 1 >= max_frames:
            print("Reached max frame limit.")
            break

    cap.release()
    out.release()
    total_time = time.time() - start_time
    print(f"Upscaling completed in {total_time:.2f} seconds.")

# 실행
if __name__ == "__main__":
    model_path = "models/RRDB_ESRGAN_x4.pth"  # 모델 경로 설정
    input_video = "C:/work_lyj/upscale_video_proj/src/video_to_upscale.mp4"
    output_video = "C:/work_lyj/upscale_video_proj/src/output_video.mp4"
    upscale_size = (1080, 810)  # 목표 해상도

    print("모델 로드 중...")
    model = load_model(model_path)

    #필요 시 실행 전 max_frames 조정, 동영상 전체 처리를 위해서는 max_frames 인자를 지우세요
    print("업스케일링 시작...")
    upscale_video(input_video, output_video, model, upscale_size, max_frames=10)
    print("완료되었습니다!")