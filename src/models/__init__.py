import torch
from .rrdbnet_arch import RRDBNet  # RRDBNet 모델 임포트

# ESRGAN 모델 로드
def load_model(model_path):
    # ESRGAN 모델 생성
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)  # 기본 설정
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()  # 평가 모드로 전환
    return model