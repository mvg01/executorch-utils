"""
벤치마크 결과 분석 및 비교 유틸리티
docs/ 폴더의 JSON 결과 파일을 로드하고 분석하는 모듈
"""

import os
import json
from typing import List, Dict, Optional
from pathlib import Path


class BenchmarkResult:
    """단일 벤치마크 결과를 나타내는 클래스"""

    def __init__(self, data: dict, file_path: str = ""):
        self.model_name = data.get("model_name", "Unknown")
        self.model_type = data.get("model_type", "Unknown")
        self.precision = data.get("precision", "Unknown")
        self.precision_short = data.get("precision_short", "unknown")
        self.delegate = data.get("delegate", "unknown")
        self.repeat = data.get("repeat", 0)
        self.warmup = data.get("warmup", 0)
        self.file_path = file_path

        latency = data.get("latency_ms", {})
        self.avg_latency = latency.get("avg", 0.0)
        self.min_latency = latency.get("min", 0.0)
        self.max_latency = latency.get("max", 0.0)
        self.std_dev = latency.get("std_dev", 0.0)
        self.p95 = latency.get("p95", 0.0)
        self.p99 = latency.get("p99", 0.0)

    @property
    def display_name(self) -> str:
        """화면 표시용 이름"""
        return f"{self.model_name}_{self.precision_short}_{self.delegate}"

    @property
    def config_key(self) -> str:
        """설정 구분용 키 (정렬/그룹핑에 사용)"""
        return f"{self.precision_short}_{self.delegate}"

    def __repr__(self):
        return f"BenchmarkResult({self.display_name}, avg={self.avg_latency:.2f}ms)"


class BenchmarkAnalyzer:
    """벤치마크 결과를 로드하고 분석하는 클래스"""

    def __init__(self, results_dir: str = "docs"):
        self.results_dir = Path(results_dir)
        self.results: List[BenchmarkResult] = []

    def load_all_results(self) -> List[BenchmarkResult]:
        """docs/ 폴더의 모든 JSON 파일을 로드"""
        self.results = []

        if not self.results_dir.exists():
            return self.results

        json_files = list(self.results_dir.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    result = BenchmarkResult(data, str(json_file))
                    self.results.append(result)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue

        return self.results

    def get_models(self) -> List[str]:
        """사용 가능한 모든 모델 이름 리스트"""
        return sorted(list(set(r.model_name for r in self.results)))

    def get_precisions(self) -> List[str]:
        """사용 가능한 모든 정밀도 타입"""
        return sorted(list(set(r.precision_short for r in self.results)))

    def get_delegates(self) -> List[str]:
        """사용 가능한 모든 델리게이트 타입"""
        return sorted(list(set(r.delegate for r in self.results)))

    def filter_by_model(self, model_name: str) -> List[BenchmarkResult]:
        """특정 모델의 모든 결과 반환"""
        return [r for r in self.results if r.model_name == model_name]

    def filter_by_precision(self, precision: str) -> List[BenchmarkResult]:
        """특정 정밀도의 모든 결과 반환"""
        return [r for r in self.results if r.precision_short == precision]

    def filter_by_delegate(self, delegate: str) -> List[BenchmarkResult]:
        """특정 델리게이트의 모든 결과 반환"""
        return [r for r in self.results if r.delegate == delegate]

    def compare_precisions(self, model_name: str, delegate: str = "xnnpack") -> Dict[str, float]:
        """
        동일 모델의 FP32 vs INT8 성능 비교

        Returns:
            {
                'fp32_avg': 10.5,
                'int8_avg': 7.2,
                'speedup': 1.46,  # fp32 / int8
                'improvement_pct': 31.4  # ((fp32 - int8) / fp32) * 100
            }
        """
        results = [r for r in self.results
                  if r.model_name == model_name and r.delegate == delegate]

        fp32_result = next((r for r in results if r.precision_short == "fp32"), None)
        int8_result = next((r for r in results if r.precision_short == "int8"), None)

        if not fp32_result or not int8_result:
            return {}

        speedup = fp32_result.avg_latency / int8_result.avg_latency if int8_result.avg_latency > 0 else 0
        improvement = ((fp32_result.avg_latency - int8_result.avg_latency) / fp32_result.avg_latency * 100) if fp32_result.avg_latency > 0 else 0

        return {
            'fp32_avg': fp32_result.avg_latency,
            'int8_avg': int8_result.avg_latency,
            'speedup': speedup,
            'improvement_pct': improvement
        }

    def compare_delegates(self, model_name: str, precision: str = "fp32") -> Dict[str, float]:
        """
        동일 모델의 XNNPACK vs Portable 성능 비교

        Returns:
            {
                'xnnpack_avg': 10.5,
                'portable_avg': 15.2,
                'speedup': 1.45
            }
        """
        results = [r for r in self.results
                  if r.model_name == model_name and r.precision_short == precision]

        xnnpack_result = next((r for r in results if r.delegate == "xnnpack"), None)
        portable_result = next((r for r in results if r.delegate == "portable"), None)

        if not xnnpack_result or not portable_result:
            return {}

        speedup = portable_result.avg_latency / xnnpack_result.avg_latency if xnnpack_result.avg_latency > 0 else 0

        return {
            'xnnpack_avg': xnnpack_result.avg_latency,
            'portable_avg': portable_result.avg_latency,
            'speedup': speedup
        }

    def get_fastest_config(self) -> Optional[BenchmarkResult]:
        """전체 결과 중 가장 빠른 설정"""
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.avg_latency)

    def get_model_ranking(self, precision: str = "fp32", delegate: str = "xnnpack") -> List[BenchmarkResult]:
        """특정 설정에서 모델들을 속도 순으로 정렬"""
        filtered = [r for r in self.results
                   if r.precision_short == precision and r.delegate == delegate]
        return sorted(filtered, key=lambda r: r.avg_latency)

    def generate_summary_stats(self) -> dict:
        """전체 벤치마크 결과의 요약 통계"""
        if not self.results:
            return {}

        return {
            'total_results': len(self.results),
            'total_models': len(self.get_models()),
            'fastest_overall': self.get_fastest_config(),
            'avg_latency_all': sum(r.avg_latency for r in self.results) / len(self.results),
            'models': self.get_models(),
            'precisions': self.get_precisions(),
            'delegates': self.get_delegates()
        }


if __name__ == "__main__":
    # 테스트 코드
    analyzer = BenchmarkAnalyzer("docs")
    results = analyzer.load_all_results()

    print(f"Loaded {len(results)} results")
    print(f"Models: {analyzer.get_models()}")
    print(f"Precisions: {analyzer.get_precisions()}")
    print(f"Delegates: {analyzer.get_delegates()}")

    if results:
        fastest = analyzer.get_fastest_config()
        print(f"\nFastest config: {fastest}")

        # MobileNetV2 FP32 vs INT8 비교 예시
        comparison = analyzer.compare_precisions("mobilenet_v2", "xnnpack")
        if comparison:
            print(f"\nMobileNetV2 XNNPACK Comparison:")
            print(f"  FP32: {comparison['fp32_avg']:.2f}ms")
            print(f"  INT8: {comparison['int8_avg']:.2f}ms")
            print(f"  Speedup: {comparison['speedup']:.2f}x")
            print(f"  Improvement: {comparison['improvement_pct']:.1f}%")