import time
import os
import torch
import torchvision.models as models
import numpy as np
import json
from torch.utils import _pytree as pytree
import traceback

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSemanticSegmentation,
    CLIPTextModel
)

from executorch.runtime import Runtime 
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig, to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# 양자화 관련
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

MODEL_CONFIGS = {
    "resnet18": {
        "type": "cv", "measure_runs": 50, "warmup_runs": 5, "assert_tolerance": 1e-5,
    },
    "mobilenet_v2": {
        "type": "cv", "measure_runs": 50, "warmup_runs": 5, "assert_tolerance": 1e-5,
    },
    "distilbert": {
        "type": "llm_encoder", "model_id_for_load": "distilbert-base-uncased",
        "measure_runs": 10, "warmup_runs": 3, "assert_tolerance": 1e-3,
    },
    "clip_text": {
        "type": "clip_text", "model_id_for_load": "openai/clip-vit-base-patch32",
        "measure_runs": 10, "warmup_runs": 3, "assert_tolerance": 1e-3,
    },
    "efficientnet_b0": {
        "type": "cv", "measure_runs": 50, "warmup_runs": 5, "assert_tolerance": 1e-5,
    },
    "segformer_b0": {
        "type": "cv_segmentation", "model_id_for_load": "nvidia/segformer-b0-finetuned-ade-512-512",
        "measure_runs": 10, "warmup_runs": 3, "assert_tolerance": 1e-3,
    },
}

def measure_latency(func, warmup_runs: int, measure_runs: int, log_callback):
    latencies = []
    log_callback(f"Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs): 
        _ = func()
    log_callback(f"Warmup complete. Starting benchmark ({measure_runs} runs)...")
    for i in range(measure_runs):
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)
        if (i + 1) % 10 == 0 or (i + 1) == measure_runs:
             log_callback(f"  Run {i+1}/{measure_runs} complete.")
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    min_latency = np.min(latencies)
    std_dev = np.std(latencies)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    return {"avg": avg_latency, "max": max_latency, "min": min_latency,
            "std_dev": std_dev, "p95": p95, "p99": p99}


def _get_calibration_inputs(model_type, model_id):
    """캘리브레이션용 입력 데이터 생성"""
    try:
        if model_type == "cv":
            return (torch.randn(1, 3, 224, 224),)
        elif model_type == "llm_encoder":
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            token_inputs = tokenizer("Sample calibration text for quantization.", return_tensors="pt")
            return (token_inputs['input_ids'], token_inputs['attention_mask'])
        elif model_type == "clip_text":
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            token_inputs = tokenizer("a photo of a cat", return_tensors="pt")
            return (token_inputs['input_ids'], token_inputs['attention_mask'])
        elif model_type == "cv_segmentation":
            return (torch.randn(1, 3, 512, 512),)
        else:
            return (torch.randn(1, 3, 224, 224),)
    except Exception:
        return (torch.randn(1, 3, 224, 224),)


def run_full_benchmark_task(
    model_name: str, 
    repeat: int,
    delegate: str, 
    precision: str,
    log_callback
) -> dict | None:

    log_callback(f"\n[Phase 1] Loading config for: {model_name}")
    if model_name not in MODEL_CONFIGS:
        log_callback(f"Error: Model '{model_name}' not found in MODEL_CONFIGS.")
        return None

    model_config = MODEL_CONFIGS[model_name]
    model_type = model_config["type"]
    model_id = model_config.get("model_id_for_load", model_name)
    warmup_runs = model_config.get("warmup_runs", 5)
    measure_runs = repeat

    precision_short_str = 'int8' if precision == 'INT8 (PT2E Quant)' else 'fp32'
    pte_path = f"temp_{model_name}_{precision_short_str}_{delegate}.pte"
    
    log_callback(f"Config loaded. Type: {model_type}, Precision: {precision}, Delegate: {delegate}")

    model = None
    example_args = None
    example_kwargs = {}
    et_inputs_list = None
    program = None # 최종 ExecuTorch Program

    try:
        # --- Phase 2: 모델 로드 및 입력 생성 ---
        log_callback(f"\n[Phase 2] Loading model and creating inputs...")

        if model_type == "cv":
            model = getattr(models, model_name)(weights="DEFAULT").eval()
            example_args = (torch.randn(1, 3, 224, 224),)
        elif model_type == "llm_encoder":
            model = AutoModel.from_pretrained(model_id).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            token_inputs = tokenizer("Hello, world!", return_tensors="pt")
            example_args = (token_inputs['input_ids'], token_inputs['attention_mask'])
        elif model_type == "clip_text":
            model = CLIPTextModel.from_pretrained(model_id).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            token_inputs = tokenizer("a photo of a cat", return_tensors="pt")
            example_args = (token_inputs['input_ids'], token_inputs['attention_mask'])
        elif model_type == "cv_segmentation":
            model = AutoModelForSemanticSegmentation.from_pretrained(model_id).eval()
            example_args = (torch.randn(1, 3, 512, 512),)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        et_inputs_list = list(example_args)
        log_callback(f"Model '{model_name}' loaded successfully.")

        # --- Phase 2.5: Quantization / FP32 Export ---
        if precision == "INT8 (PT2E Quant)":
            # --- INT8 PATH ---
            log_callback("\n[Phase 2.5] Applying INT8 Quantization (torchao PT2E)...")
            
            # Step 1: Export for training (ExecuTorch 공식)
            log_callback("  1. Exporting with export_for_training...")
            with torch.no_grad():
                training_gm = torch.export.export_for_training(model, example_args).module()
            log_callback("     Training GraphModule created.")

            # Step 2: Quantizer 설정 (per-channel for better performance)
            log_callback("  2. Setting up XNNPACKQuantizer...")
            quantizer = XNNPACKQuantizer()
            quantization_config = get_symmetric_quantization_config(
                is_per_channel=True  # 🆕 변경: per-channel 양자화로 성능 개선
            )
            quantizer.set_global(quantization_config)
            log_callback("     Quantizer configured (per-channel - optimized for speed).")

            # Step 3: Prepare 
            log_callback("  3. Preparing model (prepare_pt2e)...")
            prepared_model = prepare_pt2e(training_gm, quantizer)
            log_callback("     Model prepared with observers.")

            # Step 4: Calibration
            log_callback("  4. Running calibration (10 batches)...")
            with torch.no_grad():
                for i in range(10):
                    calib_inputs = _get_calibration_inputs(model_type, model_id)
                    try:
                        _ = prepared_model(*calib_inputs)
                        if (i + 1) % 5 == 0: log_callback(f"     Batch {i+1}/10: Complete")
                    except Exception as e:
                        log_callback(f"     Batch {i+1}/10: Warning - {str(e)[:60]}")
            log_callback("     Calibration complete.")

            # Step 5: Convert
            log_callback("  5. Converting to INT8 (convert_pt2e)...")
            quantized_model = convert_pt2e(prepared_model)
            log_callback("     INT8 conversion complete.")

            # Step 6: Final export (Quantized model)
            log_callback("  6. Final export to ExportedProgram...")
            with torch.no_grad():
                exported_program = torch.export.export(
                    quantized_model,
                    args=example_args,
                    strict=False
                )
            log_callback("     Quantized model exported.")
            
            # --- Phase 3 & 4 (INT8): 최적화된 경로 시도 ---
            log_callback(f"[Phase 3] Converting to EdgeProgram and Lowering ({delegate})...")

            # 🆕 변경: FP32와 동일한 최적화 경로 사용 (per-channel 양자화와 함께)
            if delegate == "xnnpack":
                partitioners = [XnnpackPartitioner()]
                log_callback("     Using optimized XNNPACK path for INT8...")
            else:
                partitioners = []
                log_callback("     Using Portable path for INT8...")

            try:
                # 최적화된 API 시도 (FP32와 동일)
                program = to_edge_transform_and_lower(
                    exported_program,
                    partitioner=partitioners
                ).to_executorch()
                log_callback(f"     ✓ INT8 compiled with optimized path ({delegate}).")
            except Exception as opt_err:
                # Fallback: 기존 방식
                log_callback(f"     Optimized path failed: {str(opt_err)[:50]}, using fallback...")
                edge_config = EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True)
                edge_program = to_edge(exported_program, compile_config=edge_config)
                if partitioners:
                    program = edge_program.to_backend(partitioners[0]).to_executorch()
                else:
                    program = edge_program.to_executorch()
                log_callback(f"     INT8 compiled with fallback path ({delegate}).")
                
        else:
            # --- FP32 PATH ---
            log_callback("\n[Phase 2.5] Exporting model (FP32)...")
            with torch.no_grad():
                exported_program = torch.export.export(
                    model, 
                    args=example_args,
                    strict=False
                )
            log_callback("Model exported successfully (FP32).")

            # --- Phase 3 & 4 (FP32): to_edge_transform_and_lower 사용 ---
            log_callback(f"\n[Phase 3] Converting to EdgeProgram and Lowering ({delegate})...")

            if delegate == "xnnpack":
                partitioners = [XnnpackPartitioner()]
                log_callback("     Using XNNPACK partitioner.")
            else:
                partitioners = []
                log_callback("     Using Portable partitioner.")

            # [FIXED] 최적화가 잘 되었던 'to_edge_transform_and_lower' API 사용
            program = to_edge_transform_and_lower(
                exported_program,
                partitioner=partitioners
            ).to_executorch()
            log_callback("     FP32 model converted and lowered successfully.")


        # --- Phase 6: .pte 저장 ---
        log_callback(f"\n[Phase 6] Saving compiled model to {pte_path}...")
        with open(pte_path, "wb") as f:
            f.write(program.buffer)
        file_size_kb = os.path.getsize(pte_path) / 1024
        log_callback(f".pte file created successfully ({file_size_kb:.1f} KB).")

        # --- Phase 7: 벤치마크 실행 ---
        log_callback(f"\n[Phase 7] Running benchmark...")
        
        runtime = Runtime.get()
        method = runtime.load_program(pte_path).load_method("forward")

        def executorch_run():
            return method.execute(et_inputs_list)

        latency_results = measure_latency(executorch_run, warmup_runs, measure_runs, log_callback)

        del method
        del runtime
        log_callback("Benchmark complete.")

        # --- Phase 8: 결과 포맷팅 ---
        final_results = {
            "model_name": model_name,
            "model_type": model_type,
            "precision": precision,
            "precision_short": precision_short_str,
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
        log_callback(f"\n[FATAL ERROR]")
        log_callback(f"Exception occurred: {e}")
        log_callback(traceback.format_exc())
        return None

    finally:
        if os.path.exists(pte_path):
            try:
                time.sleep(0.1)
                os.remove(pte_path)
                log_callback(f"\nTemporary file {pte_path} removed successfully.")
            except Exception as e:
                log_callback(f"Warning: Failed to remove temporary file {pte_path}: {e}")