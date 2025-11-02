import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog
import threading
import queue
import json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

try:
    import benchmark_logic
    import benchmark_analyzer
    import benchmark_visualizer
except ImportError as e:
    messagebox.showerror("Import Error", f"필요한 모듈을 찾을 수 없습니다: {e}")
    exit()


class ExecuTorchBenchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ExecuTorch Delegate Benchmark Tool - Enhanced")
        self.root.geometry("1000x700")

        self.log_queue = queue.Queue()
        self.final_results_json = None

        # Analyzer & Visualizer 초기화
        self.analyzer = benchmark_analyzer.BenchmarkAnalyzer("docs")
        self.visualizer = benchmark_visualizer.BenchmarkVisualizer(figure_size=(9, 5), dpi=90)

        # 탭 컨테이너 생성
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Tab 1: 기존 벤치마크 실행 탭
        self.benchmark_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.benchmark_tab, text="🚀 Run Benchmark")

        # Tab 2: 결과 비교 탭
        self.compare_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.compare_tab, text="📊 Compare Results")

        # 각 탭 UI 구성
        self.setup_benchmark_tab()
        self.setup_compare_tab()

    # ==================== Tab 1: 벤치마크 실행 ====================
    def setup_benchmark_tab(self):
        main_frame = ttk.Frame(self.benchmark_tab, padding="10")
        main_frame.pack(fill="both", expand=True)

        config_frame = ttk.LabelFrame(main_frame, text="Benchmark Configuration")
        config_frame.pack(fill="x", padx=5, pady=5)

        # 모델 선택
        ttk.Label(config_frame, text="Model:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(config_frame, textvariable=self.model_var, width=25, state="readonly")
        try:
            self.model_configs = benchmark_logic.MODEL_CONFIGS
            model_list = sorted(list(self.model_configs.keys()))
            self.model_combo['values'] = model_list
            if model_list: self.model_var.set(model_list[0])
        except Exception as e:
            messagebox.showerror("Config Error", f"MODEL_CONFIGS 로드 실패: {e}")
            self.model_combo['values'] = ["Error"]; self.model_var.set("Error")
        self.model_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_select)

        # 반복 횟수
        ttk.Label(config_frame, text="Repeat (Recommend):").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.repeat_var = tk.IntVar()
        self.repeat_spinbox = ttk.Spinbox(config_frame, from_=1, to=10000, textvariable=self.repeat_var, width=8)
        self.repeat_spinbox.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.on_model_select() # 초기값 설정

        ttk.Label(config_frame, text="Precision:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.precision_var = tk.StringVar(value="FP32")
        self.precision_combo = ttk.Combobox(config_frame, textvariable=self.precision_var, width=25, state="readonly")
        self.precision_combo['values'] = ["FP32", "INT8 (PT2E Quant)"]
        self.precision_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(config_frame, text="Delegate:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.delegate_var = tk.StringVar(value="xnnpack")
        ttk.Radiobutton(config_frame, text="XNNPACK", variable=self.delegate_var, value="xnnpack").grid(row=2, column=1, padx=(5, 10), pady=5, sticky="w")
        ttk.Radiobutton(config_frame, text="Portable", variable=self.delegate_var, value="portable").grid(row=2, column=1, padx=(90, 0), pady=5, sticky="w")

        config_frame.grid_columnconfigure(1, weight=1)

        # 실행 및 저장 버튼
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", padx=5, pady=(5, 10))

        self.run_button = ttk.Button(button_frame, text="🚀 Run Benchmark", command=self.start_benchmark_task)
        self.run_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.save_button = ttk.Button(button_frame, text="💾 Save Results (JSON)", command=self.save_results, state="disabled")
        self.save_button.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # 로그 출력
        log_frame = ttk.LabelFrame(main_frame, text="Logs & Results")
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_text = ScrolledText(log_frame, wrap=tk.WORD, height=20, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.process_log_queue() # 큐 모니터링 시작

    # ==================== Tab 2: 결과 비교 ====================
    def setup_compare_tab(self):
        main_frame = ttk.Frame(self.compare_tab, padding="10")
        main_frame.pack(fill="both", expand=True)

        # 상단: 컨트롤 패널
        control_frame = ttk.LabelFrame(main_frame, text="Visualization Controls")
        control_frame.pack(fill="x", padx=5, pady=5)

        # 차트 타입 선택 (ranking과 distribution 제거)
        ttk.Label(control_frame, text="Chart Type:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.chart_type_var = tk.StringVar(value="precision")
        chart_types = [
            ("FP32 vs INT8", "precision"),
            ("XNNPACK vs Portable", "delegate"),
            ("INT8 Speedup", "speedup")
        ]

        for idx, (text, value) in enumerate(chart_types):
            ttk.Radiobutton(control_frame, text=text, variable=self.chart_type_var,
                          value=value, command=self.update_chart).grid(row=0, column=idx+1, padx=5, pady=5)

        # 버튼들 (필터 제거, Refresh Data만 유지)
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=6, sticky="ew", padx=5, pady=5)

        ttk.Button(button_frame, text="🔄 Refresh Data", command=self.refresh_results).pack(side="left", padx=5)
        ttk.Button(button_frame, text="💾 Export Chart", command=self.export_chart).pack(side="left", padx=5)

        # 중단: 차트 표시 영역
        self.chart_frame = ttk.LabelFrame(main_frame, text="Visualization")
        self.chart_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # 초기 데이터 로드
        self.refresh_results()

    def refresh_results(self):
        """docs/ 폴더에서 결과 다시 로드"""
        try:
            results = self.analyzer.load_all_results()

            if not results:
                messagebox.showwarning("No Data", "docs/ 폴더에 벤치마크 결과가 없습니다.\n먼저 벤치마크를 실행해주세요.")
                return

            # 차트 업데이트
            self.update_chart()

            messagebox.showinfo("Success", f"{len(results)}개의 결과를 로드했습니다.")

        except Exception as e:
            messagebox.showerror("Error", f"결과 로드 중 오류 발생:\n{e}")

    def update_chart(self):
        """선택된 차트 타입에 따라 차트 업데이트"""
        # 기존 차트 제거
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        try:
            results = self.analyzer.results
            if not results:
                label = ttk.Label(self.chart_frame, text="No data available. Please run benchmarks first.",
                                font=("Arial", 12))
                label.pack(expand=True)
                return

            # 차트 타입에 따라 생성
            chart_type = self.chart_type_var.get()

            if chart_type == "precision":
                # FP32 vs INT8 비교 (xnnpack 고정)
                fig = self.visualizer.create_precision_comparison_chart(results, "xnnpack")
            elif chart_type == "delegate":
                # XNNPACK vs Portable 비교 (fp32 고정)
                fig = self.visualizer.create_delegate_comparison_chart(results, "fp32")
            elif chart_type == "speedup":
                # INT8 Speedup (xnnpack 고정)
                xnnpack_results = [r for r in results if r.delegate == "xnnpack"]
                fig = self.visualizer.create_speedup_chart(xnnpack_results)
            else:
                return

            # Tkinter에 차트 임베드
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

            # 현재 차트 저장 (export 용)
            self.current_figure = fig

        except Exception as e:
            messagebox.showerror("Chart Error", f"차트 생성 중 오류 발생:\n{e}")
            import traceback
            traceback.print_exc()

    def export_chart(self):
        """현재 차트를 이미지 파일로 저장"""
        if not hasattr(self, 'current_figure'):
            messagebox.showwarning("No Chart", "저장할 차트가 없습니다.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Chart",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )

        if file_path:
            try:
                self.current_figure.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"차트가 {file_path}에 저장되었습니다.")
            except Exception as e:
                messagebox.showerror("Export Error", f"차트 저장 중 오류:\n{e}")

    # ==================== 기존 벤치마크 기능 (유지) ====================
    def on_model_select(self, event=None):
        try:
            model_name = self.model_var.get()
            if model_name in self.model_configs:
                recommended_runs = self.model_configs[model_name].get("measure_runs", 20)
                self.repeat_var.set(recommended_runs)
            else: self.repeat_var.set(20)
        except Exception: self.repeat_var.set(20)

    def start_benchmark_task(self):
        self.run_button.config(text="Running... 🏃‍♂️", state="disabled")
        self.save_button.config(state="disabled")
        self.log_text.config(state="normal"); self.log_text.delete(1.0, tk.END); self.log_text.config(state="disabled")
        self.final_results_json = None

        model = self.model_var.get()
        repeat = self.repeat_var.get()
        delegate = self.delegate_var.get()
        precision = self.precision_var.get()

        if model == "Error":
             messagebox.showerror("Error", "모델 설정을 불러오지 못했습니다.")
             self.run_button.config(text="Run Benchmark", state="normal")
             return

        threading.Thread(
            target=self.run_benchmark_in_thread,
            args=(model, repeat, delegate, precision),
            daemon=True
        ).start()

    def run_benchmark_in_thread(self, model, repeat, delegate, precision):
        def thread_safe_log(message): self.log_queue.put(message)
        try:
            final_result = benchmark_logic.run_full_benchmark_task(
                model, repeat, delegate, precision, thread_safe_log
            )
            if final_result:
                self.final_results_json = final_result
                result_json_str = json.dumps(final_result, indent=2)
                self.log_queue.put("\n--- Benchmark Success --- ")
                self.log_queue.put("--- Final Results (JSON) ---")
                self.log_queue.put(result_json_str)
            else:
                self.log_queue.put("\n--- Benchmark Failed ---")
                self.log_queue.put("Logs를 확인하여 원인을 분석하세요.")
        except Exception as e:
            self.log_queue.put(f"\n--- FATAL ERROR ---")
            self.log_queue.put(f"An unexpected error occurred: {e}")
            import traceback; self.log_queue.put(traceback.format_exc())
        finally: self.log_queue.put("---TASK_COMPLETE---")

    def process_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                if message == "---TASK_COMPLETE---":
                    self.run_button.config(text="Run Benchmark", state="normal")
                    if self.final_results_json: self.save_button.config(state="normal")
                else: self.log_to_gui(message)
        except queue.Empty: pass
        self.root.after(100, self.process_log_queue)

    def log_to_gui(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, str(message) + "\n")
        self.log_text.configure(state="disabled")
        self.log_text.see(tk.END)

    def save_results(self):
        if not self.final_results_json:
            messagebox.showwarning("No Data", "저장할 벤치마크 결과가 없습니다.")
            return

        model_name = self.final_results_json.get('model_name', 'results')
        precision_str = self.final_results_json.get('precision_short', 'fp32')
        delegate_name = self.final_results_json.get('delegate', 'unknown')

        initial_filename = f"{model_name}_{precision_str}_{delegate_name}.json"

        file_path = filedialog.asksaveasfilename(
            title="Save Benchmark Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=initial_filename,
            initialdir="docs"  # docs 폴더를 기본 저장 위치로
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.final_results_json, f, indent=2)
                messagebox.showinfo("Success", f"결과가 {file_path}에 성공적으로 저장되었습니다.")

                # 저장 후 Compare 탭 자동 새로고침 제안
                if messagebox.askyesno("Refresh", "Compare Results 탭을 새로고침하시겠습니까?"):
                    self.refresh_results()

            except Exception as e:
                messagebox.showerror("Save Error", f"파일 저장 중 오류가 발생했습니다:\n{e}")


if __name__ == "__main__":
    try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    root = tk.Tk()
    app = ExecuTorchBenchGUI(root)
    root.mainloop()
