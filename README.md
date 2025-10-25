PyTorch 모델을 ExecuTorch(`.pte`)로 변환하고, 지정된 횟수만큼 추론을 실행하여 성능(Latency)을 측정하는 벤치마크 도구입니다. CLI와 GUI 두 가지 인터페이스를 제공합니다. docs 디렉터리에 json 파일로 저장한 결과 파일이 있습니다.

## 구현 기능 요약

### 백엔드 (`benchmark_logic.py`)

- **동적 모델 로드**: `torchvision.models` 또는 `transformers` 라이브러리에서 `resnet18`, `mobilenet_v2`, `distilbert`, `segformer_b0` 등 다양한 모델을 문자열 이름으로 로드합니다.
  - 컴퓨터 비전 (CV): `resnet18`, `mobilenet_v2`, `efficientnet_b0`
  - 분할 (Segmentation): `segformer_b0`
  - 트랜스포머/LLM 인코더: `distilbert`, `clip_text`
- **정밀도 선택 (FP32/INT8)**:
  - **FP32**: `torch.export` 및 `to_edge_transform_and_lower` API를 사용하여 FP32 모델을 `.pte` 파일로 자동 변환합니다.
  - **INT8 (PTQ)**: ExecuTorch의 `XNNPACKQuantizer`를 활용한 Post-Training Quantization(학습 후 양자화) 기능을 구현했습니다. `per-tensor` 방식을 사용하여 런타임 호환성을 확보했습니다.
- **델리게이트 지원**: ExecuTorch 백엔드로 `XNNPACK` (CPU 최적화) 또는 `Portable` (범용)을 선택하여 컴파일 및 실행할 수 있습니다. (INT8 양자화는 XNNPACK 델리게이트에 최적화되어 있습니다.)
- **벤치마킹**: 지정된 `repeat` 횟수만큼 추론을 실행하여 평균/최소/최대 Latency(ms), 표준편차, p95/p99 백분위수 등 상세 성능 지표를 측정합니다.
- **자동 파일 관리**: 생성된 임시 `temp_*.pte` 파일은 벤치마크 완료 후 자동으로 삭제됩니다.
- **상태 콜백**: `print` 대신 콜백 함수를 사용하여 로깅하므로 CLI, GUI 등 다양한 프론트엔드와 연동 가능합니다.

### CLI 인터페이스 (`run_bench.py`)

- 기존 과제 요구사항을 충족하는 명령줄 인터페이스입니다.
- `--model`, `--repeat`, `--delegate` 등의 인자를 받아 벤치마크를 실행하고 결과를 콘솔 또는 `--output` 파일(JSON)로 출력합니다.
- **(참고)** 현재 `run_bench.py`는 `benchmark_logic.py`와 동기화되지 않았으며, 초기 버전의 로직을 포함하고 있습니다. 최신 기능과 모델 지원은 GUI 버전을 사용해주세요.


### GUI 인터페이스 (`benchmark_gui.py`)

- Tkinter를 사용하여 개발된 사용자 친화적인 그래픽 인터페이스입니다.
- 정밀도 선택: `FP32`와 `INT8 (PT2E Quant)`를 선택할 수 있는 드롭다운 메뉴를 제공합니다.
- 모델 선택 드롭다운 (지원 목록 자동 로드), 반복 횟수 설정 (모델별 추천값 제공), 델리게이트 선택 (라디오 버튼) 기능을 제공합니다.
- `threading`과 `queue`를 사용하여 백그라운드에서 벤치마크를 실행하므로, 컴파일이나 실행 중에도 GUI가 멈추지 않습니다.
- 실행 로그와 최종 결과(JSON)를 실시간으로 텍스트 창에 표시합니다.
- 벤치마크 결과를 JSON 파일로 저장하는 기능을 제공하며, 파일명에 모델과 델리게이트 정보(`모델명_정밀도_델리게이트명.json`)를 자동으로 포함합니다.



## 개발 환경

- **Python**: `3.10.19` 
- 필수 패키지 (자세한 버전은 `requirements.txt` 참조)
  - torch==2.9.0
  - torchvision==0.24.0
  - executorch==1.0.0
  - pytest==8.4.2
  - numpy==2.2.6
  - transformers==4.57.1
- 실행 환경: x64 Native Tools Command Prompt for VS 2022 (관리자 권한)



## 설치 및 실행 방법

1. **Repo 복제**

   ```bash
   git clone https://github.com/mvg01/executorch-utils.git
   cd executorch-utils
   ```

2. **Visual Studio 설치**: Visual Studio 2022와 "C++를 사용한 데스크톱 개발" 워크로드를 설치

3. **Python 환경 준비** (Conda 권장)

   ```bash
   conda create -n exec_project python=3.10
   conda activate exec_project
   ```

4. **라이브러리 설치**

   ```bash
   pip install -r requirements.txt
   ```

5. **실행**

   - **GUI 버전 (권장)**

     ```bash
     python benchmark_gui.py
     ```

   - CLI 버전 (델리게이트 XNNPACK 고정)

     ```bash
     # 실행 예시
     python run_bench.py --model resnet18 --repeat 5
     ```

   - 테스트 코드 실행

     ```
     pytest -v -s test_run_bench.py
     ```



## 실행 결과 예시

### GUI 프로그램 실행

<img width="977" height="873" alt="image" src="https://github.com/user-attachments/assets/48edea7a-8533-40e4-abc2-7cac5f9617fc" />

### CLI 프로그램 실행

<img width="1075" height="285" alt="image" src="https://github.com/user-attachments/assets/c33da63b-85e6-48e6-be66-458eddd163a9" />

### Pytest (`test_run_bench.py`) - 참고용

<img width="1081" height="226" alt="image" src="https://github.com/user-attachments/assets/ca284b07-2146-45e7-803c-c4521e07e28d" />



## 현재 이슈 및 제한 사항

1. **(해결됨) INT8 양자화 지원**
    - **내용**: 안정적인 FP32 벤치마크 기능에 더해, INT8 PTQ(Post-Training Quantization) 기능이 성공적으로 구현되었습니다.
    - **기술적 해결**: 구현 초기, `per-channel` 양자화 방식(`is_per_channel=True`)이 ExecuTorch 런타임에서 `Missing out variants: {'quantized_decomposed::dequantize_per_channel'}` 오류를 일으키는 호환성 문제를 발견했습니다.
    - **해결**: 이는 런타임 호환성이 더 높은 **`per-tensor`** (`is_per_channel=False`) 양자화로 변경하고, `EdgeCompileConfig`에 **`_skip_dim_order=True`** 플래그를 추가하여 해결했습니다.
    - **결과**: 사용자는 이제 GUI에서 FP32와 INT8의 성능을 직접 비교할 수 있습니다. (예: `mobilenet_v2` + `XNNPACK` 기준, INT8이 FP32 대비 약 1.4배의 속도 향상 및 높은 안정성을 보임)
2. **`gpt2` 모델 Export 실패**:
   - `gpt2` 모델은 FP32 Export 단계 (`torch.export.export` 호출 시)에서 `AttributeError: 'FakeTensor' object has no attribute 'get_seq_length'` 오류가 발생하며 실패했습니다.
   - 이는 `gpt2` 모델 내부 구현이 `torch.export`의 트레이싱 방식(FakeTensor 사용)과 호환되지 않기 때문으로 분석됩니다.
   - 결론: `gpt2` 모델 지원은 현재 버전에서 **제외**되었습니다.
3. **`segformer_b0` 모델 + `Portable` 델리게이트 성능 이슈**:
   - `segformer_b0` 모델을 FP32 정밀도로 벤치마킹할 때, `XNNPACK` 델리게이트에서는 정상적인 속도로 실행되지만, `Portable` 델리게이트를 사용하면 실행 속도가 극도로 느려지거나 응답이 없는 현상이 관찰되었습니다.
   - 이는 `Portable` 백엔드가 `segformer_b0`과 같이 복잡하고 입력 크기가 큰 모델에 대해 최적화 수준이 낮기 때문으로 추정됩니다.
   - 결론: 이 현상은 코드 버그라기보다는 ExecuTorch 백엔드의 성능 특성을 보여주는 결과로 판단되며, 해결되지 않은 관찰 사항으로 기록합니다. 이는 XNNPACK과 같은 최적화된 델리게이트 사용의 중요성을 보여줍니다.
