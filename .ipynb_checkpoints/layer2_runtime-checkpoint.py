"""
Layer 2: Runtime Hooks — 适配 vLLM 0.23.0 v1 架构
"""
import time
from .core import BUS, TraceEvent

# 调试开关，开启后打印调度/执行次数，排查问题时打开
DEBUG = False
_schedule_count = 0
_execute_count = 0

def _extract_rids(sched_output) -> list[str]:
    """从SchedulerOutput中提取请求ID列表，转成字符串统一格式"""
    tokens = getattr(sched_output, "num_scheduled_tokens", {}) or {}
    rids = list(tokens.keys())
    return [str(r) for r in rids] if rids else ["batch"]

def _is_prefill(sched_output) -> bool:
    """判断本轮是否包含Prefill阶段（有新请求进入）"""
    new_reqs = getattr(sched_output, "scheduled_new_reqs", None) or []
    return len(new_reqs) > 0

class RuntimeHooks:
    def install(self):
        """安装所有Runtime层Hook，必须在LLM实例化前调用"""
        self._hook_scheduler()
        self._hook_model_runner()
        print("[vllm-tracer] Runtime hooks已安装 (vLLM 0.23.0 v1 单进程模式)")

    def _hook_scheduler(self):
        """Hook调度器的schedule方法，每轮调度触发一次"""
        from vllm.v1.core.sched.scheduler import Scheduler

        orig_schedule = Scheduler.schedule

        def wrapped_schedule(self_sched, *args, **kwargs):
            global _schedule_count
            _schedule_count += 1

            t_start = time.perf_counter_ns()
            output = orig_schedule(self_sched, *args, **kwargs)
            dur = time.perf_counter_ns() - t_start

            rids = _extract_rids(output)
            token_map = getattr(output, "num_scheduled_tokens", {}) or {}
            total_tokens = getattr(output, "total_num_scheduled_tokens", 0)
            finished = getattr(output, "finished_req_ids", set()) or set()

            # 全局调度事件（每个调度周期一个，放在main轨道）
            BUS.emit(TraceEvent(
                name="schedule_step",
                layer="runtime",
                cat="scheduler",
                request_id=None,
                ts_ns=t_start,
                dur_ns=dur,
                meta={
                    "batch_size": len(rids),
                    "total_tokens": total_tokens,
                    "finished_count": len(finished),
                    "step": _schedule_count
                }
            ))

            # 每个请求单独记录调度事件
            for rid in rids:
                BUS.emit(TraceEvent(
                    name="schedule",
                    layer="runtime",
                    cat="scheduler",
                    request_id=rid,
                    ts_ns=t_start,
                    dur_ns=dur,
                    meta={
                        "tokens": token_map.get(rid, 0),
                        "finished": rid in finished,
                        "step": _schedule_count
                    }
                ))

            if DEBUG:
                print(f"[调试] 第{_schedule_count}次调度，batch_size={len(rids)}，总token={total_tokens}")

            return output

        Scheduler.schedule = wrapped_schedule

    def _hook_model_runner(self):
        """Hook模型执行器的execute_model方法，每轮模型执行触发一次"""
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        orig_exec = GPUModelRunner.execute_model

        def wrapped_exec(self_mr, scheduler_output, intermediate_tensors=None, **kwargs):
            global _execute_count
            _execute_count += 1

            phase = "prefill" if _is_prefill(scheduler_output) else "decode"
            t_start = time.perf_counter_ns()
            out = orig_exec(self_mr, scheduler_output, intermediate_tensors, **kwargs)
            dur = time.perf_counter_ns() - t_start

            rids = _extract_rids(scheduler_output)
            token_map = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
            total_tokens = getattr(scheduler_output, "total_num_scheduled_tokens", 0)

            # 全局执行事件
            BUS.emit(TraceEvent(
                name=f"{phase}_batch",
                layer="runtime",
                cat=phase,
                request_id=None,
                ts_ns=t_start,
                dur_ns=dur,
                meta={
                    "batch_size": len(rids),
                    "total_tokens": total_tokens,
                    "phase": phase,
                    "step": _execute_count
                }
            ))

            # 每个请求单独记录执行事件
            for rid in rids:
                BUS.emit(TraceEvent(
                    name=phase,
                    layer="runtime",
                    cat=phase,
                    request_id=rid,
                    ts_ns=t_start,
                    dur_ns=dur,
                    meta={
                        "tokens": token_map.get(rid, 0),
                        "phase": phase,
                        "step": _execute_count
                    }
                ))

            if DEBUG:
                print(f"[调试] 第{_execute_count}次执行，phase={phase}，batch_size={len(rids)}")

            return out

        GPUModelRunner.execute_model = wrapped_exec
