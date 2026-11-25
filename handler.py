import os
import runpod
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_ID = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "50000"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))

# Load once when the worker process starts; requests reuse the model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
llm = LLM(
    model=MODEL_ID,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    trust_remote_code=True,
)

def handler(job):
    data = job.get("input", {})
    messages = data.get("messages")
    prompt = data.get("prompt")

    if messages:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if not prompt or not isinstance(prompt, str):
        return {"error": "Provide input.prompt or input.messages."}

    params = SamplingParams(
        temperature=float(data.get("temperature", 0.2)),
        top_p=float(data.get("top_p", 0.95)),
        max_tokens=int(data.get("max_tokens", 1024)),
    )
    result = llm.generate([prompt], params)
    return {
        "text": result[0].outputs[0].text,
        "model": MODEL_ID,
        "usage": {
            "prompt_tokens": len(result[0].prompt_token_ids),
            "completion_tokens": len(result[0].outputs[0].token_ids),
        },
    }

runpod.serverless.start({"handler": handler})
