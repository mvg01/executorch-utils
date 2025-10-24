모델 이름을 인자로 받아, 동적으로 `torchvision`에서 모델을 로드하고 `.pte`로 변환하여 벤치마크를 수행하는 CLI 도구(`run_bench.py`)입니다.



### 구현 기능 요약

- **동적 모델 로드**: `torchvision.models`에서 `getattr`을 사용해 `resnet18`, `mobilenet_v2` 등 다양한 모델을 문자열 이름으로 로드합니다.
- **자동 변환 및 최적화**: `torch.export` 및 `XnnpackPartitioner`를 사용해 모델을 `.pte` 파일로 자동 변환 및 최적화합니다.
- **벤치마킹**: 지정된 `repeat` 횟수만큼 추론을 실행하여 평균 Latency(ms)를 측정합니다.
- **JSON 출력**: 벤치마크 결과를 `model_name`, `latency_ms_avg`, `repeat`가 포함된 JSON 형식으로 출력합니다.
- **자동 파일 관리**: 생성된 임시 `temp_*.pte` 파일은 벤치마크 완료 후 자동으로 삭제됩니다.



### 개발 환경

- **Python**: `3.10.19`
- 필수 패키지
  - torch==2.9.0
  - torchvision==0.24.0
  - executorch==1.0.0
  - pytest==8.4.2
  - numpy==2.2.6
- 실행 환경: x64 Native Tools Command Prompt for VS 2022



### 설치 및 실행 방법

1. repo 복제

```bash
git clone https://github.com/YourUsername/executorch-benchmark.git
cd executorch-benchmark
```

2. Visual Studio 2022와 "C++를 사용한 데스크톱 개발" 워크로드를 설치
3. Python 환경 준비 (Conda)

```bash
conda create -n exec_project python=3.10
conda activate exec_project
```

4. 라이브러리 설치 (pip)

```bash
pip install -r requirements.txt
```

5. 실행

   - x64 Native Tools Command Prompt for VS 2022 터미널 **관리자 권한**으로 실행

   - ```bash
     # 실행 스크립트 예시
     python run_bench.py --model resnet18 --repeat 5
     
     # run_bench가 제대로 작동하는지 검증하는 테스트 실행
     pytest -v -s
     ```



### 실행 결과 예시

run_bench.py
<img width="1075" height="285" alt="image" src="https://github.com/user-attachments/assets/c33da63b-85e6-48e6-be66-458eddd163a9" />


pytest
<img width="1081" height="226" alt="image" src="https://github.com/user-attachments/assets/ca284b07-2146-45e7-803c-c4521e07e28d" />



