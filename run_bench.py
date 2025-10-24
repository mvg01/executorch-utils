import argparse
import time
import json
import os
import torch
import torchvision.models as models
import numpy as np

from executorch.runtime import Runtime
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

def get_model_name(file_path: str) -> str:
    base_name = os.path.basename(file_path)
    model_name, _ = os.path.splitext(base_name)
    return model_name

def get_or_create_pte(model_arg: str) -> (str, bool):
    print(f" {model_arg} 모델을 torchvision.models에서 동적으로 로드 시도... ")
    temp_pte_path = f"temp_{model_arg}.pte"
    
    try:
        # 문자열 이름(model_arg)으로 models.resnet18 같은 함수를 찾음
        model_func = getattr(models, model_arg)
        
        # "DEFAULT" 가중치로 모델 로드
        model = model_func(weights="DEFAULT").eval()
        
        # 대부분의 torchvision 비전 모델은 이 입력을 사용한다고 가정.
        example_inputs = (torch.randn(1, 3, 224, 224),)

    except AttributeError:
        print(f"오류: 'torchvision.models'에 {model_arg} 라는 이름의 모델이 없습니다.")
        return None, False
    except TypeError as e:
        print(f"오류: {model_arg} 모델 로드 중 오류 발생 (weights 인자 문제일 수 있음): {e}")
        return None, False
    except Exception as e:
        print(f"오류: {model_arg} 모델 로드 중 알 수 없는 오류: {e}")
        return None, False

    # .pte 변환 과정
    try:
        print(f"{model_arg} 모델 .pte로 변환 시작...")
        exported_program = torch.export.export(model, example_inputs)
        program = to_edge_transform_and_lower(
            exported_program,
            partitioner=[XnnpackPartitioner()]
        ).to_executorch()
        
        with open(temp_pte_path, "wb") as f:
            f.write(program.buffer)
        
        print(f"{temp_pte_path} 생성 완료")
        return temp_pte_path, True
        
    except Exception as e:
        print(f"오류: {model_arg} 모델을 .pte로 변환 중 에러 발생: {e}")
        if os.path.exists(temp_pte_path):
            os.remove(temp_pte_path)
        return None, False

# 모델의 latency 측정
def run_benchmark(model_path: str, repeat: int):
    runtime = Runtime.get() 
    try:
        method = runtime.load_program(model_path).load_method("forward")
        print(f"{model_path} 로드 성공")
    except Exception as e:
        print(f"오류: 모델 로드 중 에러 발생: {e}")
        return None

    dummy_input = [torch.randn(1, 3, 224, 224)]
    latencies_ms = [] 
    
    try:
        _ = method.execute(dummy_input) 
        print(f"워밍업 1회 실행 완료. 본 벤치마크({repeat}회)를 시작합니다...")

        # 사용자가 요청한 repeat 횟수만큼 반복
        for _ in range(repeat):
            start_time = time.perf_counter()
            _ = method.execute(dummy_input)  # 모델 추론 실행
            end_time = time.perf_counter() 
            latencies_ms.append((end_time - start_time) * 1000)
            
    except Exception as e:
        print(f"오류: 모델 실행 중 에러 발생: {e}")
        return None 

    avg_latency = np.mean(latencies_ms)  # Latency 리스트의 평균값 계산
    
    results = {
        "model_name": get_model_name(model_path).replace("temp_", ""), # "temp_" 접두어 제거
        "latency_ms_avg": round(avg_latency, 2), 
        "repeat": repeat
    }
    
    return results

def main():
    # CLI 인자 파서(parser) 생성 및 설정
    parser = argparse.ArgumentParser(
        description="ExecuTorch 모델 벤치마크 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter, 
        epilog="""
        사용 예시:
        python %(prog)s --model resnet18 --repeat 10
        python %(prog)s --model mobilenet_v2 --repeat 5
        """
    )
    
    # '--model' 인자 정의 (필수)
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="벤치마크할 모델 이름(예: resnet18, mobilenet_v2)"
    )
    
    # '--repeat' 인자 정의 (필수)
    parser.add_argument(
        "--repeat", 
        type=int, 
        required=True, 
        help="추론을 반복할 횟수"
    )
    
    # 사용자가 입력한 CLI 인자 파싱
    args = parser.parse_args()

    # .pte 파일 생성 시도 (모델 이름 인자 전달)
    pte_path, was_created = get_or_create_pte(args.model)
    
    if pte_path is None:
        print("벤치마크 중단")
        return

    # .pte 파일로 벤치마크 실행
    results_data = run_benchmark(pte_path, args.repeat)

    if was_created:
        try:
            os.remove(pte_path) 
            print(f"임시 파일 {pte_path} 삭제 완료")
        except Exception as e:
            print(f"경고: 임시 파일 {pte_path} 삭제 실패: {e}")

    # 벤치마크 결과(딕셔너리)가 정상적으로 반환되었으면 JSON으로 출력
    if results_data:
        print("\n벤치마크 결과 (JSON)")
        print(json.dumps(results_data, indent=2))

if __name__ == "__main__":
    main()