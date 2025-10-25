import time
import os
import torch
import torchvision.models as models
import numpy as np
import json
from torch.utils import _pytree as pytree

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSemanticSegmentation,
    CLIPTextModel
)

from executorch.runtime import Runtime
from executorch.exir import (
    to_edge_transform_and_lower,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


MODEL_CONFIGS = {
    "resnet18": {
        "type": "cv", "measure_runs": 50, "warmup_runs": 10, "assert_tolerance": 1e-5,
    },
    "mobilenet_v2": {
        "type": "cv", "measure_runs": 50, "warmup_runs": 10, "assert_tolerance": 1e-5,
    },
    "distilbert": {
        "type": "llm_encoder", "model_id_for_load": "distilbert-base-uncased",
        "measure_runs": 10, "warmup_runs": 5, "assert_tolerance": 1e-3,
    },
    "clip_text": {
        "type": "clip_text", "model_id_for_load": "openai/clip-vit-base-patch32",
        "measure_runs": 30, "warmup_runs": 5, "assert_tolerance": 1e-3,
    },
    "efficientnet_b0": {
        "type": "cv", "measure_runs": 50, "warmup_runs": 10, "assert_tolerance": 1e-5,
    },
    "segformer_b0": {
        "type": "cv_segmentation", "model_id_for_load": "nvidia/segformer-b0-finetuned-ade-512-512",
        "measure_runs": 10, "warmup_runs": 2, "assert_tolerance": 1e-3,
    },
}

def measure_latency(func, warmup_runs: int, measure_runs: int, status_callback):
    latencies = []
    status_callback(f"Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs): _ = func()
    status_callback(f"Warmup complete. Starting benchmark ({measure_runs} runs)...")
    for i in range(measure_runs):
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)
        if (i + 1) % 10 == 0 or (i + 1) == measure_runs:
             status_callback(f"  Run {i+1}/{measure_runs} complete.")
    avg_latency = np.mean(latencies); max_latency = np.max(latencies)
    min_latency = np.min(latencies); std_dev = np.std(latencies)
    p95 = np.percentile(latencies, 95); p99 = np.percentile(latencies, 99)
    return {"avg": avg_latency, "max": max_latency, "min": min_latency,
            "std_dev": std_dev, "p95": p95, "p99": p99}


# GUI가 호출할 메인 함수 
def run_full_benchmark_task(model_name: str, repeat_from_gui: int, delegate: str, status_callback) -> dict | None:

    status_callback(f"\n[Phase 1] Loading config for: {model_name}")
    if model_name not in MODEL_CONFIGS:
        status_callback(f"Error: Model '{model_name}' not found in MODEL_CONFIGS.")
        return None

    model_config = MODEL_CONFIGS[model_name]
    model_type = model_config["type"]
    model_id = model_config.get("model_id_for_load", model_name)
    warmup_runs = 5
    measure_runs = repeat_from_gui

    precision = "fp32"
    pte_path = f"temp_{model_name}_{precision}_{delegate}.pte"
    status_callback(f"Config loaded. Type: {model_type}, Precision: {precision}, Delegate: {delegate}")

    model = None
    example_args = None
    example_kwargs = None
    et_inputs_list = None

    try:
        # 모델 로드 및 입력 생성
        status_callback(f"\n[Phase 2] Loading model and creating inputs...")

        if model_type == "cv":
            model = getattr(models, model_name)(weights="DEFAULT").eval()
            example_args = (torch.randn(1, 3, 224, 224),)
            example_kwargs = {}
        elif model_type == "llm_encoder":
            model = AutoModel.from_pretrained(model_id).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            token_inputs = tokenizer("Hello, world!", return_tensors="pt")
            example_args = (token_inputs['input_ids'], token_inputs['attention_mask'])
            example_kwargs = {}
        elif model_type == "clip_text":
            model = CLIPTextModel.from_pretrained(model_id).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            token_inputs = tokenizer("a photo of a cat", return_tensors="pt")
            example_args = (token_inputs['input_ids'], token_inputs['attention_mask'])
            example_kwargs = {}
        elif model_type == "cv_segmentation":
            model = AutoModelForSemanticSegmentation.from_pretrained(model_id).eval()
            example_args = (torch.randn(1, 3, 512, 512),)
            example_kwargs = {}
        else: raise ValueError(f"Unsupported model type: {model_type}")

        et_inputs_list = list(example_args) + list(example_kwargs.values())
        status_callback(f"Model '{model_name}' loaded successfully.")

        status_callback(f"\n[Phase 3] Compiling/Lowering model (Precision: {precision})...") # Phase 번호 조정

        if delegate == "xnnpack":
            partitioners = [XnnpackPartitioner()]
        elif delegate == "portable":
            partitioners = []
        else:
            status_callback(f"Warning: Unknown delegate '{delegate}'. Using portable.")
            partitioners = []

        status_callback("Using 'torch.export' + 'to_edge_transform_and_lower' (FP32 Path)")

        # Export
        exported_program = torch.export.export(model, args=example_args, kwargs=example_kwargs)
        status_callback("Model exported successfully.")

        # Compile
        program = to_edge_transform_and_lower(
            exported_program,
            partitioner=partitioners
        ).to_executorch()


        status_callback(f"Saving compiled model to {pte_path}...")
        with open(pte_path, "wb") as f:
            f.write(program.buffer)
        status_callback(f".pte file created successfully.")


        # 벤치마크 실행
        status_callback(f"\n[Phase 4] Running benchmark...")
        runtime = Runtime.get()
        method = runtime.load_program(pte_path).load_method("forward")

        def executorch_run():
            return method.execute(et_inputs_list)

        latency_results = measure_latency(executorch_run, warmup_runs, measure_runs, status_callback)

        del method
        del runtime
        status_callback("Benchmark complete.")

        # 결과 포맷팅
        final_results = {
            "model_name": model_name,
            "model_type": model_type,
            "precision": precision,
            "delegate": delegate,
            "repeat": measure_runs,
            "warmup": warmup_runs,
            "latency_ms": {
                "avg": round(latency_results["avg"], 2),
                "min": round(latency_results["min"], 2),
                "max": round(latency_results["max"], 2),
                "std_dev": round(latency_results["std_dev"], 2),
                "p95": round(latency_results["p95"], 2),
                "p99": round(latency_results["p99"], 2),
            }
        }
        return final_results

    except Exception as e:
        status_callback(f"\nFATAL ERROR")
        status_callback(f"An exception occurred: {e}")
        import traceback
        status_callback(traceback.format_exc())
        return None

    finally:
        if os.path.exists(pte_path):
            try:
                time.sleep(0.1)
                os.remove(pte_path)
                status_callback(f"\nTemporary file {pte_path} removed successfully.")
            except Exception as e:
                status_callback(f"Warning: Failed to remove temporary file {pte_path}: {e}")