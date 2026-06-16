from .core import BUS
from .layer1_kernel import KernelProfiler
from .layer2_runtime import RuntimeHooks
from .layer3_serving import ServingAnalyzer

class VLLMTracer:
    def __init__(self):
        self.kernel_profiler = KernelProfiler()
        self.runtime_hooks = RuntimeHooks()
        self.serving_analyzer = ServingAnalyzer()

    def setup(self):
        """安装所有Hook，必须在实例化LLM之前调用"""
        self.runtime_hooks.install()

    def start_profiler(self):
        """启动全层性能采集，在推理开始前调用"""
        self.kernel_profiler.start()

    def finish(self, output_path: str = "vllm_trace.json"):
        """停止采集，计算指标，导出Trace文件"""
        # 停止Kernel采集
        self.kernel_profiler.stop()
        # 计算Serving层指标
        metrics = self.serving_analyzer.compute()
        # 导出Trace文件
        BUS.export_chrome_trace(output_path)
        return metrics
