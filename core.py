import time
import json
import threading
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TraceEvent:
    name: str
    layer: str
    cat: str = ""
    request_id: Optional[str] = None
    ts_ns: int = 0
    dur_ns: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.cat:
            self.cat = self.layer

class EventBus:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.events = []
                cls._instance.t0 = time.perf_counter_ns()
                cls._instance.cuda_time_offset_ns = 0
        return cls._instance

    def set_cuda_time_offset(self, offset_ns: int):
        self.cuda_time_offset_ns = offset_ns

    def emit(self, event: TraceEvent):
        with self._lock:
            self.events.append(event)

    def span(self, name, layer, cat="", request_id=None, **meta):
        return _Span(self, name, layer, cat, request_id, meta)

    def export_chrome_trace(self, path: str = "vllm_trace.json"):
        pid_map = {
            "kernel":  1,
            "runtime": 2,
            "serving": 3,
        }
        pid_name_map = {
            1: "Layer 1: GPU Kernel",
            2: "Layer 2: Runtime (调度/执行)",
            3: "Layer 3: Serving (请求生命周期)",
        }

        trace_events = []

        # 进程元数据
        for pid, name in pid_name_map.items():
            trace_events.append({
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": name},
            })

        # ✅ 导出所有事件
        for e in self.events:
            pid = pid_map.get(e.layer, 9)

            # kernel层按GPU设备分轨道，避免同轨道时间重叠
            if e.layer == "kernel":
                device_idx = e.meta.get("device_index", 0)
                tid = f"gpu_{device_idx}"
            else:
                tid = f"req_{e.request_id}" if e.request_id else "main"

            trace_events.append({
                "name": e.name,
                "cat":  e.cat,
                "ph":   "X",
                "ts":   (e.ts_ns - self.t0) / 1000,  # 纳秒→微秒
                "dur":  max(e.dur_ns / 1000, 0.001),  # 防止dur=0被Perfetto忽略
                "pid":  pid,
                "tid":  tid,
                "args": e.meta,
            })

        with open(path, "w") as f:
            json.dump({"traceEvents": trace_events}, f)

        print(f"[vllm-tracer] Trace已导出: {path}，共{len(trace_events)}个事件")
        print(f"[vllm-tracer] 其中实际数据事件: {len(self.events)}个，元数据事件: {len(pid_name_map)}个")


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
            meta=self.meta,
        ))

BUS = EventBus()
