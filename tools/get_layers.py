from ultralytics import YOLO
import torch
from torchinfo import summary

# 1. 대상 모델 로드 (Teacher 모델로 사용할 사전 학습된 가중치)
# 워크스테이션 환경이므로 GPU로 바로 로드할 수 있습니다.
model = YOLO("yolov8n-pose.pt").to("cuda")

# YOLO 클래스 내부에 감싸져 있는 실제 PyTorch nn.Module 객체를 꺼냅니다.
pytorch_model = model.model 

print("[1] Forward Hook 타겟용 정확한 레이어 이름 추출")
# 모든 모듈을 순회하며 이름과 타입을 출력합니다.
for name, module in pytorch_model.named_modules():
    # 출력이 너무 길어지는 것을 방지하기 위해, 
    # 피처맵 추출의 주요 타겟이 되는 Conv2d나 핵심 블록(C2f 등)만 필터링해서 봅니다.
    if isinstance(module, torch.nn.Conv2d) or "C2f" in module.__class__.__name__:
        print(f"Layer Name: {name:<20} | Type: {module.__class__.__name__}")

print("\n")
print("[2] Adapter 설계를 위한 각 레이어별 텐서 차원(Shape) 확인")
# 더미 입력값(1장, 3채널, 640x640 해상도)을 통과시켜
# 각 레이어에서 피처맵 채널이 어떻게 변하는지 확인합니다.
summary(pytorch_model, input_size=(1, 3, 640, 640), device="cuda")