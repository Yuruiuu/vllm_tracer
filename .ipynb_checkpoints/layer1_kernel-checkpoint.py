import time
import torch
from torch.profiler import profile, ProfilerActivity
from torch.autograd import DeviceType
from .core import BUS, TraceEvent


class KernelProfiler:
    def __init__(self):
        self.profiler = None
        self.kernel_count = 0
        self.t_start_cpu_ns = 0

    def start(self):
        """启动Kernel采集，同时记录CPU时间锚点完成对齐"""
        torch.cuda.synchronize()
        self.t_start_cpu_ns = time.perf_counter_ns()

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
        torch.cuda.synchronize()
        self.profiler.__exit__(None, None, None)

        # 用 DeviceType.CUDA 枚举过滤
        kernel_events = [
            evt for evt in self.profiler.events()
            if evt.device_type == DeviceType.CUDA
        ]

        print(f"[vllm-tracer][debug] 过滤到 {len(kernel_events)} 个CUDA Kernel事件")

        for evt in kernel_events:
            ts_ns  = self.t_start_cpu_ns + int(evt.time_range.start * 1000)
            dur_ns = int(evt.time_range.elapsed_us() * 1000)

            BUS.emit(TraceEvent(
                name=evt.name,
                layer="kernel",
                cat="cuda_kernel",
                request_id=None,
                ts_ns=ts_ns,
                dur_ns=dur_ns,
                meta={
                    "device_index": evt.device_index,        # ✅ int，GPU编号
                    "is_async":     evt.is_async,            # ✅ bool，是否异步
                    "scope":        evt.scope,               # ✅ int，0=fwd,1=bwd
                }
            ))

        self.kernel_count = len(kernel_events)
        print(f"[vllm-tracer] KernelProfiler采集完成，共{self.kernel_count}个CUDA Kernel事件")

