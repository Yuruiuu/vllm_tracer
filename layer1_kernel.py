import time
import torch
from torch.profiler import profile, ProfilerActivity
from .core import BUS, TraceEvent

class KernelProfiler:
    def __init__(self):
        self.profiler = None
        # profiler启动瞬间的CPU绝对时间戳（纳秒），用于时间对齐
        self.t_start_cpu_ns = 0

    def start(self):
        """启动Kernel采集，同时记录CPU时间锚点完成对齐"""
        # 先同步CUDA，清空待执行任务，保证时间锚点准确
        torch.cuda.synchronize()
        
        # 记录profiler启动时的CPU绝对时间，作为时间对齐基准
        self.t_start_cpu_ns = time.perf_counter_ns()

        # 启动PyTorch Profiler，同时采集CPU和CUDA事件
        self.profiler = profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False
        )
        self.profiler.__enter__()
        print("[vllm-tracer] KernelProfiler已启动，CPU-CUDA时间基准已对齐")

    def stop(self):
        """停止采集，解析Kernel事件并写入全局事件总线"""
        # 同步所有CUDA操作，确保所有Kernel执行完毕再停止采集
        torch.cuda.synchronize()
        self.profiler.__exit__(None, None, None)

        # 过滤出纯CUDA Kernel事件
        kernel_events = [
            evt for evt in self.profiler.events()
            if evt.device_type == torch.device("cuda").type
        ]
        
        # 转换事件格式，对齐时间基准后写入BUS
        for evt in kernel_events:
            # 相对时间 + CPU基准时间 = 绝对CPU时间（纳秒），和Runtime层完全对齐
            ts_ns = int(evt.time_range.start.item()) + self.t_start_cpu_ns
            # 持续时间从微秒转换为纳秒
            dur_ns = int(evt.time_range.elapsed_us().item() * 1000)

            BUS.emit(TraceEvent(
                name=evt.name,
                layer="kernel",
                cat="cuda_kernel",
                request_id=None,
                ts_ns=ts_ns,
                dur_ns=dur_ns,
                meta={
                    "device": str(evt.device),
                    "kernel_type": evt.kind,
                }
            ))

        self.kernel_count = len(kernel_events)
        print(f"[vllm-tracer] KernelProfiler采集完成，共{self.kernel_count}个CUDA Kernel事件")
