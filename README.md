# yolov8-image
https://colab.research.google.com/drive/13TvEYcJAAt76440OcTTyQ42EysgNFnwj#scrollTo=WoAiWY9UPS6r&line=9&uniqifier=1
# 1단계: 필요한 라이브러리 설치
print("Ultralytics 라이브러리를 설치합니다...")
!pip install ultralytics -q
print("설치 완료.")
print("-" * 30) # 구분선

# 필요한 모듈 import
import os
from ultralytics import YOLO
from IPython.display import display, Image
import cv2 # OpenCV 사용

# 2단계: YOLOv8n 모델 파일 다운로드
model_path = '/content/yolov8x.pt'
print(f"YOLOv8n 모델 파일 ({model_path}) 확인 및 다운로드를 시작합니다...")
if not os.path.exists(model_path):
    print(f"{model_path} 파일을 다운로드합니다...")
    # wget 명령어를 사용하여 Ultralytics 저장소에서 yolov8n.pt 파일을 다운로드합니다.
    !wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt -O {model_path}
    print("모델 파일 다운로드 완료.")
else:
    print(f"{model_path} 파일이 이미 존재합니다. 다운로드를 건너뜁니다.")
print("-" * 30) # 구분선

# 3단계: 입력 이미지 파일 경로 설정 (사용자가 '/content/1.jpg'를 업로드했다고 가정)
image_path = '/content/sample_data/1.jpg.jpg'
output_path = '/content/output.jpg' # 결과를 저장할 경로


# 입력 이미지 파일 존재 여부 확인
if not os.path.exists(image_path):
    print(f"오류: 입력 이미지 파일 '{image_path}'를 찾을 수 없습니다.")
    print("Colab의 /content/ 디렉토리에 '1.jpg' 파일을 업로드했는지 확인해주세요.")
else:
    print(f"입력 이미지 파일: {image_path}")
    print(f"결과 저장 경로: {output_path}")
    print("-" * 30) # 구분선

    # 4단계: 객체 인식 수행 및 결과 저장/표시
    print("YOLOv8n 모델을 로드합니다...")
    model = YOLO(model_path)
    print("모델 로드 완료.")

    print(f"'{image_path}' 이미지에 대해 객체 인식을 수행합니다...")
    results = model(image_path)
    print("객체 인식 완료.")

    # 결과 처리 및 저장
    print("결과 이미지를 생성하고 저장합니다...")
    annotated_image = results[0].plot()
    cv2.imwrite(output_path, annotated_image)
    print(f"결과 이미지를 '{output_path}'에 저장했습니다.")
    print("-" * 30) # 구분선

    # 저장된 결과 이미지를 Colab 노트북에 표시합니다.
    print("결과 이미지를 표시합니다:")
    display(Image(filename=output_path))
