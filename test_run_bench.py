import pytest
import os
import subprocess
import json

# resnet18 같은 모델 이름을 인자로 줬을 때, CLI가 .pte 파일을 자동으로 생성, 실행, 삭제하는지 검증
def test_run_bench_with_model_name():
    model_name = "resnet18"
    repeat_count = 3
    temp_file_path = f"temp_{model_name}.pte"
    
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    result = subprocess.run(
        [
            "python", 
            "run_bench.py", 
            "--model", model_name, 
            "--repeat", str(repeat_count)
        ],
        capture_output=True, text=True, check=True
    )

    try:
        json_output = result.stdout[result.stdout.find('{'):]
        output_data = json.loads(json_output)
    except json.JSONDecodeError:
        pytest.fail(f"CLI 출력이 유효한 JSON이 아닙니다. 출력:\n{result.stdout}")

    assert output_data["model_name"] == model_name
    assert output_data["repeat"] == repeat_count
    assert output_data["latency_ms_avg"] > 0
    assert not os.path.exists(temp_file_path), \
        f"스크립트가 임시 파일({temp_file_path})을 삭제하지 않았습니다."


# 지원하지 않는 모델 이름을 주는 경우
def test_run_bench_with_unsupported_model_name():
    """
    (시나리오 2) 지원하지 않는 모델 이름(예: "unknown_model")을 줬을 때,
    스크립트가 오류 메시지를 출력하고 JSON을 출력하지 않는지 검증합니다.
    """
    model_name = "unknown_test"
    repeat_count = 3

    result = subprocess.run(
        [
            "python", 
            "run_bench.py", 
            "--model", model_name, 
            "--repeat", str(repeat_count)
        ],
        capture_output=True, text=True
    )
    assert "{" not in result.stdout, "오류가 발생했는데 JSON이 출력되었습니다."
    assert "벤치마크 결과" not in result.stdout
    assert f"오류: 'torchvision.models'에 {model_name} 라는 이름의 모델이 없습니다." in result.stdout
    assert "벤치마크 중단" in result.stdout