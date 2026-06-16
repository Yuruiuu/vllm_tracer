import time
import json
import threading
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TraceEvent:
    name: str                      # 事件名
    layer: str                     # kernel / runtime / serving
    cat: str = ""                  # 事件分类，用于Perfetto染色，默认等于layer
    request_id: Optional[str] = None  # 关联的请求ID
    ts_ns: int = 0                 # 开始时间（纳秒，CPU基准）
    dur_ns: int = 0                # 持续时长（纳秒）
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.cat:
            self.cat = self.layer

class EventBus:
    """全局单例，三层共享，线程安全"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.events = []
                cls._instance.t0 = time.perf_counter_ns()  # CPU时间基准零点
                cls._instance.cuda_time_offset_ns = 0       # CUDA→CPU时间偏移量
        return cls._instance

    def set_cuda_time_offset(self, offset_ns: int):
        """设置CUDA时间到CPU时间的偏移量：CPU时间 = CUDA时间 + offset_ns"""
        self.cuda_time_offset_ns = offset_ns

    def emit(self, event: TraceEvent):
        with self._lock:
            self.events.append(event)

    def span(self, name, layer, cat="", request_id=None, **meta):
        """上下文管理器：自动记录开始/结束"""
        return _Span(self, name, layer, cat, request_id, meta)

    def export_chrome_trace(self, path: str = "vllm_trace.json"):
        """导出为Perfetto / chrome://tracing 兼容的格式"""
        # layer → pid映射，三层分开展示
        pid_map = {
            "kernel": 1,
            "runtime": 2,
            "serving": 3
        }
        pid_name_map = {
            1: "Layer 1: GPU Kernel",
            2: "Layer 2: Runtime (调度/执行)",
            3: "Layer 3: Serving (请求生命周期)"
        }

        trace_events = []

        # 添加进程元数据，让Perfetto显示进程名称
        for pid, name in pid_name_map.items():
            trace_events.append({
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": name}
            })

        # 导出所有事件
        for e in self.events:
            pid = pid_map.get(e.layer, 9)
            # tid命名优化：全局事件用main，请求事件用req_xxx
            tid = f"req_{e.request_id}" if e.request_id else "main"

            trace_events.append({
                "name": e.name,
                "cat": e.cat,
                "ph": "X",  # 完整持续事件
                "ts": (e.ts_ns - self.t0) / 1000,  # 转微秒，相对于零点
                "dur": e.dur_ns / 1000,
                "pid": pid,
                "tid": tid,
                "args": e.meta
            })

        with open(path, "w") as f:
            json.dump({"traceEvents": trace_events}, f)
        print(f"[vllm-tracer] Trace已导出: {path}，共{len(trace_events)}个事件")

class _Span:
    def __init__(self, bus, name, layer, cat, request_id, meta):
        self.bus = bus
        self.name = name
        self.layer = layer
        self.cat = cat
        self.request_id = request_id
        self.meta = meta

    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        dur = time.perf_counter_ns() - self.start
        self.bus.emit(TraceEvent(
            name=self.name,
            layer=self.layer,
            cat=self.cat,
            request_id=self.request_id,
            ts_ns=self.start,
            dur_ns=dur,
            meta=self.meta
        ))

# 全局单例
BUS = EventBus()
