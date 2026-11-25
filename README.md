# GPT-OSS-20B RunPod Serverless Worker

Custom Python worker using RunPod Serverless + vLLM.

## Repository files
- `handler.py` - loads the model once and handles queued inference jobs
- `requirements.txt` - Python dependencies
- `Dockerfile` - builds the worker image

## RunPod settings
- Endpoint type: Serverless / Queue-based
- Container source: Build from this GitHub repository
- Dockerfile path: `Dockerfile`
- Model: `openai/gpt-oss-20b`
- Environment variables:
  - `MODEL_NAME=openai/gpt-oss-20b`
  - `MAX_MODEL_LEN=50000`
  - `GPU_MEMORY_UTILIZATION=0.90`
- Worker maximum: 2
- Idle timeout: 5 minutes

## Request example
Submit a RunPod job with:

```json
{
  "input": {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a SQL query to select all customers."}
    ],
    "max_tokens": 500,
    "temperature": 0.2
  }
}
```

The handler returns JSON containing `text`, `model`, and token usage.

## Notes
- GPT-OSS-20B is a large model. Choose a GPU with sufficient VRAM for model weights and KV cache. A 16 GB GPU is generally insufficient for a full-precision deployment.
- First startup downloads model weights and may take time. Configure Hugging Face credentials as a secret only if your selected model access requires them.
- Check RunPod's current base-image and vLLM compatibility before production deployment. Pin compatible versions after validating the build.
