"""
벤치마크 결과 시각화 모듈
matplotlib를 사용하여 다양한 비교 차트 생성
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import List, Optional
from benchmark_analyzer import BenchmarkResult


# 한글 폰트 설정 (Windows 기준)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
except:
    pass  # 폰트 없으면 기본 폰트 사용


class BenchmarkVisualizer:
    """벤치마크 결과를 시각화하는 클래스"""

    def __init__(self, figure_size=(10, 6), dpi=100):
        self.figure_size = figure_size
        self.dpi = dpi

        # 색상 팔레트
        self.colors = {
            'fp32': '#3498db',      # 파란색
            'int8': '#e74c3c',      # 빨간색
            'xnnpack': '#2ecc71',   # 초록색
            'portable': '#f39c12',  # 주황색
            'primary': '#3498db',
            'secondary': '#9b59b6'
        }

    def create_precision_comparison_chart(self, results: List[BenchmarkResult], delegate: str = "xnnpack") -> plt.Figure:
        """
        FP32 vs INT8 비교 막대 그래프

        Args:
            results: 비교할 결과 리스트
            delegate: 델리게이트 필터 (기본값: xnnpack)
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # 델리게이트 필터링
        filtered = [r for r in results if r.delegate == delegate]

        # 모델별로 그룹핑
        models = sorted(list(set(r.model_name for r in filtered)))

        fp32_values = []
        int8_values = []
        valid_models = []

        for model in models:
            model_results = [r for r in filtered if r.model_name == model]
            fp32 = next((r.avg_latency for r in model_results if r.precision_short == "fp32"), None)
            int8 = next((r.avg_latency for r in model_results if r.precision_short == "int8"), None)

            if fp32 is not None and int8 is not None:
                valid_models.append(model)
                fp32_values.append(fp32)
                int8_values.append(int8)

        if not valid_models:
            ax.text(0.5, 0.5, 'No data available for comparison',
                   ha='center', va='center', fontsize=12)
            return fig

        # 🆕 무거운 모델 분리 (200ms 이상만 제외)
        HEAVY_THRESHOLD = 200.0
        heavy_models = []
        light_models = []
        light_fp32 = []
        light_int8 = []

        for i, model in enumerate(valid_models):
            if fp32_values[i] > HEAVY_THRESHOLD or int8_values[i] > HEAVY_THRESHOLD:
                heavy_models.append((model, fp32_values[i], int8_values[i]))
            else:
                light_models.append(model)
                light_fp32.append(fp32_values[i])
                light_int8.append(int8_values[i])

        # 가벼운 모델만 차트로 표시
        if not light_models:
            ax.text(0.5, 0.5, 'All models are heavy (>200ms)\nSee summary below',
                   ha='center', va='center', fontsize=12)
            return fig

        x = np.arange(len(light_models))
        width = 0.35

        bars1 = ax.bar(x - width/2, light_fp32, width, label='FP32',
                      color=self.colors['fp32'], alpha=0.8)
        bars2 = ax.bar(x + width/2, light_int8, width, label='INT8',
                      color=self.colors['int8'], alpha=0.8)

        # 값 레이블 추가
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}ms', ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}ms', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')

        # 무거운 모델 정보 추가
        title = f'FP32 vs INT8 Performance Comparison ({delegate.upper()})'
        if heavy_models:
            heavy_info = ', '.join([f'{m}: {fp32:.1f}/{int8:.1f}ms'
                                   for m, fp32, int8 in heavy_models])
            title += f'\n(Heavy models excluded: {heavy_info})'

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(light_models, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        return fig

    def create_delegate_comparison_chart(self, results: List[BenchmarkResult], precision: str = "fp32") -> plt.Figure:
        """
        XNNPACK vs Portable 비교 막대 그래프

        Args:
            results: 비교할 결과 리스트
            precision: 정밀도 필터 (기본값: fp32)
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # 정밀도 필터링
        filtered = [r for r in results if r.precision_short == precision]

        # 모델별로 그룹핑
        models = sorted(list(set(r.model_name for r in filtered)))

        xnnpack_values = []
        portable_values = []
        valid_models = []

        for model in models:
            model_results = [r for r in filtered if r.model_name == model]
            xnnpack = next((r.avg_latency for r in model_results if r.delegate == "xnnpack"), None)
            portable = next((r.avg_latency for r in model_results if r.delegate == "portable"), None)

            if xnnpack is not None and portable is not None:
                valid_models.append(model)
                xnnpack_values.append(xnnpack)
                portable_values.append(portable)

        if not valid_models:
            ax.text(0.5, 0.5, 'No data available for comparison',
                   ha='center', va='center', fontsize=12)
            return fig

        # 🆕 무거운 모델 분리 (2500ms 이상만 제외)
        HEAVY_THRESHOLD = 2500.0
        heavy_models = []
        light_models = []
        light_xnnpack = []
        light_portable = []

        for i, model in enumerate(valid_models):
            if xnnpack_values[i] > HEAVY_THRESHOLD or portable_values[i] > HEAVY_THRESHOLD:
                heavy_models.append((model, xnnpack_values[i], portable_values[i]))
            else:
                light_models.append(model)
                light_xnnpack.append(xnnpack_values[i])
                light_portable.append(portable_values[i])

        # 가벼운 모델만 차트로 표시
        if not light_models:
            ax.text(0.5, 0.5, 'All models are heavy (>2500ms)\nSee summary below',
                   ha='center', va='center', fontsize=12)
            return fig

        x = np.arange(len(light_models))
        width = 0.35

        bars1 = ax.bar(x - width/2, light_xnnpack, width, label='XNNPACK',
                      color=self.colors['xnnpack'], alpha=0.8)
        bars2 = ax.bar(x + width/2, light_portable, width, label='Portable',
                      color=self.colors['portable'], alpha=0.8)

        # 값 레이블 추가
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}ms', ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}ms', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')

        # 무거운 모델 정보 추가
        title = f'XNNPACK vs Portable Delegate Comparison ({precision.upper()})'
        if heavy_models:
            heavy_info = ', '.join([f'{m}: {xnn:.1f}/{port:.1f}ms'
                                   for m, xnn, port in heavy_models])
            title += f'\n(Heavy models excluded: {heavy_info})'

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(light_models, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        return fig

    def create_model_ranking_chart(self, results: List[BenchmarkResult], top_n: int = 10) -> plt.Figure:
        """
        전체 모델 성능 순위 (수평 막대 그래프)

        Args:
            results: 순위를 매길 결과 리스트
            top_n: 표시할 상위 모델 수
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        if not results:
            ax.text(0.5, 0.5, 'No data available',
                   ha='center', va='center', fontsize=12)
            return fig

        # 평균 latency 기준 정렬 (빠른 순)
        sorted_results = sorted(results, key=lambda r: r.avg_latency)[:top_n]

        labels = [r.display_name for r in sorted_results]
        values = [r.avg_latency for r in sorted_results]

        # 색상 지정 (precision에 따라)
        colors = [self.colors.get(r.precision_short, self.colors['primary'])
                 for r in sorted_results]

        y_pos = np.arange(len(labels))

        bars = ax.barh(y_pos, values, color=colors, alpha=0.8)

        # 값 레이블 추가
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value, i, f' {value:.2f}ms',
                   va='center', ha='left', fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()  # 상위 항목을 위에 표시
        ax.set_xlabel('Average Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {len(labels)} Fastest Configurations',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()
        return fig

    def create_latency_distribution_chart(self, results: List[BenchmarkResult]) -> plt.Figure:
        """
        선택된 결과들의 latency 분포 (box plot)

        Args:
            results: 분포를 표시할 결과 리스트
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        if not results:
            ax.text(0.5, 0.5, 'No data available',
                   ha='center', va='center', fontsize=12)
            return fig

        labels = [r.display_name for r in results]

        # Box plot용 데이터 구성 (min, p25, avg, p75, max 근사치)
        data = []
        for r in results:
            # Box plot을 위한 분포 근사 (실제 raw data가 없으므로 통계값으로 근사)
            data.append([r.min_latency, r.avg_latency, r.max_latency])

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                       showmeans=True, meanline=True)

        # Box plot 색상 설정
        for patch, result in zip(bp['boxes'], results):
            patch.set_facecolor(self.colors.get(result.precision_short, self.colors['primary']))
            patch.set_alpha(0.6)

        ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Latency Distribution (Min, Avg, Max)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        return fig

    def create_speedup_chart(self, results: List[BenchmarkResult], baseline: str = "fp32") -> plt.Figure:
        """
        Baseline 대비 speedup 막대 그래프

        Args:
            results: 비교할 결과 리스트
            baseline: 기준이 되는 precision (기본값: fp32)
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # 모델별 + delegate별로 그룹핑 (같은 delegate끼리만 비교)
        model_delegates = sorted(list(set((r.model_name, r.delegate) for r in results)))

        speedups = []
        valid_models = []

        for model_name, delegate in model_delegates:
            # 같은 모델 + 같은 delegate의 결과만 필터링
            model_results = [r for r in results
                           if r.model_name == model_name and r.delegate == delegate]

            baseline_result = next((r for r in model_results if r.precision_short == baseline), None)
            int8_result = next((r for r in model_results if r.precision_short == "int8"), None)

            # FP32와 INT8 둘 다 있어야 비교 가능
            if baseline_result and int8_result and int8_result.avg_latency > 0:
                speedup = baseline_result.avg_latency / int8_result.avg_latency
                speedups.append(speedup)
                valid_models.append(model_name)  # 모델명만 표시 (delegate는 필터로 선택됨)

        if not valid_models:
            ax.text(0.5, 0.5, f'No INT8 data available for {baseline.upper()} comparison',
                   ha='center', va='center', fontsize=12)
            return fig

        x = np.arange(len(valid_models))

        # 1.0보다 크면 초록, 작으면 빨강
        colors = [self.colors['xnnpack'] if s >= 1.0 else self.colors['int8']
                 for s in speedups]

        bars = ax.bar(x, speedups, color=colors, alpha=0.8)

        # 1.0 기준선
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (1.0x)')

        # 값 레이블 추가
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
        ax.set_title(f'INT8 Speedup vs {baseline.upper()} Baseline',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_models, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # 테스트 코드
    from benchmark_analyzer import BenchmarkAnalyzer

    analyzer = BenchmarkAnalyzer("docs")
    results = analyzer.load_all_results()

    if results:
        visualizer = BenchmarkVisualizer()

        # 1. FP32 vs INT8 비교
        fig1 = visualizer.create_precision_comparison_chart(results, "xnnpack")
        plt.show()

        # 2. 모델 순위
        fig2 = visualizer.create_model_ranking_chart(results, top_n=10)
        plt.show()

        # 3. Speedup 차트
        fig3 = visualizer.create_speedup_chart(results)
        plt.show()