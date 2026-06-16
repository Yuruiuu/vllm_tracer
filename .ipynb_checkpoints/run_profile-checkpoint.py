# run_profile.py
import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
import sys
import json
import torch
import torch.distributed as dist

# ========== 必须在导入vLLM之前设置环境变量 ==========
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # 强制单进程，Hook才能生效
# 已移除无效环境变量 VLLM_ATTENTION_BACKEND / VLLM_DISABLE_USAGE_STATS，避免警告

sys.path.insert(0, "/root")

from vllm_tracer import VLLMTracer
from vllm import LLM, SamplingParams

def main():
    # 安装Runtime层Hook（必须在LLM实例化之前执行）
    tracer = VLLMTracer()
    tracer.setup()

    # 实例化LLM引擎
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        quantization="awq",
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        enforce_eager=True,  # Profiler必须开启eager模式，关闭torch.compile
    )

    # 启动全层性能采集
    tracer.start_profiler()

    # 执行推理任务
    prompts = [
        "介绍一下 Transformer 的注意力机制",
        "用 Python 实现快速排序",
        "什么是 PagedAttention？",
        "解释 KV Cache 的作用",
    ]

    sampling_params = SamplingParams(max_tokens=128, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params)

    # 停止采集，导出Trace文件，计算性能指标
    metrics = tracer.finish()

    # 打印性能汇总
    print("\n" + "="*60)
    print("📊 vLLM 三层性能剖析结果")
    print("="*60)
    print(f"总请求数: {metrics['summary']['num_requests']}")
    print(f"平均首Token延迟 (TTFT): {metrics['summary']['avg_TTFT_ms']:.2f} ms")
    print(f"平均端到端延迟: {metrics['summary']['avg_E2E_ms']:.2f} ms")
    print(f"系统吞吐量: {metrics['summary']['throughput_req_s']:.2f} req/s")
    print("\n📝 单请求详情:")
    for rid, info in metrics["per_request"].items():
        print(f"  请求{rid}: E2E={info['E2E_ms']:.2f}ms | TTFT={info['TTFT_ms']:.2f}ms | 输出{info['total_output_tokens']}token")

    # 打印生成结果预览
    print("\n💬 生成结果预览:")
    for i, out in enumerate(outputs):
        text = out.outputs[0].text[:80].replace("\n", " ")
        print(f"  [{i}] {text}...")

if __name__ == "__main__":
    try:
        main()
    finally:
        # 程序退出前手动销毁分布式进程组，彻底消除资源泄漏警告
        if dist.is_initialized():
            dist.destroy_process_group()
            print("\n[vllm-tracer] 分布式进程组已正常销毁")
