import json, os, types
from .core import TraceEvent, BUS

_SINK_DIR = os.path.expanduser("~/.vllm_tracer_events")
os.makedirs(_SINK_DIR, exist_ok=True)

def install_file_sink():
    sink_path = os.path.join(_SINK_DIR, f"events_{os.getpid()}.jsonl")
    _fh = open(sink_path, 'a')
    _orig = BUS.emit

    def _patched(event: TraceEvent):
        _orig(event)
        _fh.write(json.dumps({
            'name': event.name, 'layer': event.layer,
            'request_id': event.request_id,
            'ts_ns': event.ts_ns, 'dur_ns': event.dur_ns,
            'meta': event.meta,
        }) + '\n')
        _fh.flush()

    BUS.emit = _patched
    print(f"[vllm-tracer] file sink → {sink_path}", flush=True)

def collect_all_events():
    count = 0
    for fname in sorted(os.listdir(_SINK_DIR)):
        if not fname.endswith('.jsonl'):
            continue
        fpath = os.path.join(_SINK_DIR, fname)
        with open(fpath) as f:
            for line in f:
                try:
                    BUS.emit(TraceEvent(**json.loads(line)))
                    count += 1
                except Exception:
                    pass
        os.remove(fpath)
    print(f"[vllm-tracer] 合并事件总数: {count}")
