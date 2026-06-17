"""
Layer 3: Serving层 — 请求生命周期统计与事件埋点
"""
import time
from collections import defaultdict
from .core import BUS, TraceEvent

class ServingAnalyzer:
    def compute(self):
        """计算全量请求的性能指标，同时生成Serving层事件"""
        # 按请求ID归集Runtime层事件
        by_req_events = defaultdict(list)
        for e in BUS.events:
            if e.layer == "runtime" and e.request_id:
                by_req_events[e.request_id].append(e)

        per_request = {}
        all_ttft_ms = []
        all_e2e_ms = []

        for rid, events in by_req_events.items():
            if not events:
                continue

            # 按时间排序
            events.sort(key=lambda x: x.ts_ns)

            # 分离各阶段事件
            prefill_events = [e for e in events if e.name == "prefill"]
            decode_events = [e for e in events if e.name == "decode"]
            first_schedule_ts = min(e.ts_ns for e in events)
            last_event_end_ts = max(e.ts_ns + e.dur_ns for e in events)

            # 计算TTFT：首次调度开始 → 首次Prefill结束
            ttft_ms = 0.0
            if prefill_events:
                first_prefill = prefill_events[0]
                ttft_ns = (first_prefill.ts_ns + first_prefill.dur_ns) - first_schedule_ts
                ttft_ms = ttft_ns / 1e6

            # 计算Decode相关指标
            total_decode_dur_ns = sum(e.dur_ns for e in decode_events)
            total_decode_tokens = sum(e.meta.get("tokens", 0) for e in decode_events)
            tpot_ms = (total_decode_dur_ns / 1e6) / total_decode_tokens if total_decode_tokens > 0 else 0.0

            # E2E总时长
            e2e_ms = (last_event_end_ts - first_schedule_ts) / 1e6

            per_request[rid] = {
                "TTFT_ms": round(ttft_ms, 2),
                "TPOT_ms": round(tpot_ms, 4),
                "E2E_ms": round(e2e_ms, 2),
                "decode_steps": len(decode_events),
                "total_output_tokens": total_decode_tokens
            }

            all_ttft_ms.append(ttft_ms)
            all_e2e_ms.append(e2e_ms)

            # ========== 核心新增：写入Serving层事件 ==========
            BUS.emit(TraceEvent(
                name="request_e2e",
                layer="serving",
                cat="request",
                request_id=rid,
                ts_ns=first_schedule_ts,
                dur_ns=int(last_event_end_ts - first_schedule_ts),
                meta={
                    "TTFT_ms": round(ttft_ms, 2),
                    "E2E_ms": round(e2e_ms, 2),
                    "output_tokens": total_decode_tokens
                }
            ))

        # 汇总统计
        num_requests = len(per_request)
        avg_ttft = round(sum(all_ttft_ms) / num_requests, 2) if num_requests > 0 else 0
        avg_e2e = round(sum(all_e2e_ms) / num_requests, 2) if num_requests > 0 else 0
        total_e2e_s = sum(all_e2e_ms) / 1000 if num_requests > 0 else 0
        throughput_req_s = round(num_requests / total_e2e_s, 2) if total_e2e_s > 0 else 0

        summary = {
            "num_requests": num_requests,
            "avg_TTFT_ms": avg_ttft,
            "avg_E2E_ms": avg_e2e,
            "throughput_req_s": throughput_req_s
        }

        return {"summary": summary, "per_request": per_request}
