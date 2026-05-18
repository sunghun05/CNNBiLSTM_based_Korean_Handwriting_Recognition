from ultralytics import YOLO
import onnx

# 1. 모델 로드 
# (만약 3개 클래스로 파인튜닝을 완료하셨다면 "yolov8n-pose.pt" 대신 "runs/pose/train/weights/best.pt"를 로드해야 합니다.)
model = YOLO("yolov8n-pose.pt")

# 2. Ultralytics 내장 함수로 안전하게 ONNX 변환
# 내부적으로 더미 입력 생성, 구조 최적화 등을 알아서 처리해 줍니다.
print("ONNX 변환을 시작합니다...")
onnx_file_path = model.export(
    format="onnx",
    imgsz=640,          # YOLO 기본 해상도
    dynamic=True,       # 배치 사이즈 동적 할당 허용
    opset=13,           # 타겟 하드웨어/NPU에 맞는 opset 버전
    simplify=True       # 모델 구조 최적화 (NPU 배포 시 권장)
)

print(f"변환 완료: {onnx_file_path}")

# 3. ONNX 모델 구조 검증
onnx_model = onnx.load(onnx_file_path)
onnx.checker.check_model(onnx_model)
print("ONNX 모델 검증 성공!")