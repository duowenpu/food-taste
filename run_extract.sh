#!/usr/bin/env bash
# Full pipeline runner. Steps: server -> (prepare) -> extract in waves -> aggregate.
# Usage: ./run_extract.sh [server|prepare|extract|aggregate|all]
set -euo pipefail
cd "$(dirname "$0")"
IMG=nvcr.io/nvidia/vllm:26.04-py3
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}          # host dir holding the HF hub cache
VLLM_CACHE=${VLLM_CACHE:-$HOME/.cache/vllm-compile}     # persisted torch.compile cache (faster restarts)
export FEWSHOT_IDS=${FEWSHOT_IDS:-0,1,2,4,5}   # short prompt: 5 examples, no glossary -> ~1365 tokens (1 cached block of 1072)
export NO_GLOSSARY=${NO_GLOSSARY:-1}
PY="docker run --rm --network host -e FEWSHOT_IDS -e NO_GLOSSARY -v $PWD:/work -w /work --entrypoint python3 $IMG"

server() {
  docker rm -f food-vllm >/dev/null 2>&1 || true
  docker run -d --name food-vllm --restart=no --gpus all --ipc=host --shm-size 8g -p 127.0.0.1:8092:8092 \
    -v "$HF_CACHE":/hf -v "$VLLM_CACHE":/root/.cache -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
    --entrypoint vllm $IMG serve Qwen/Qwen3.6-35B-A3B-FP8 --host 0.0.0.0 --port 8092 \
    --max-model-len 8192 --max-num-seqs 512 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.80 --trust-remote-code \
    --reasoning-parser qwen3 --enable-prefix-caching --kv-cache-dtype fp8 --mamba-cache-dtype float16 --mamba-ssm-cache-dtype float16 \
    --structured-outputs-config '{"backend":"guidance","disable_any_whitespace":true}'
  echo "waiting for health (first start ~7 min, cached ~5)"; until curl -sf http://127.0.0.1:8092/health >/dev/null; do sleep 5; done; echo healthy
}
prepare()   { $PY prepare_data.py; }
extract()   { $PY extract.py --in data/reviews.sorted.jsonl --out out/reviews.labels.jsonl --wave --concurrency 512 --log-every 2048 "$@"; }
aggregate() { $PY aggregate.py; }

case "${1:-all}" in
  server) server ;; prepare) prepare ;; extract) shift; extract "$@" ;; aggregate) aggregate ;;
  all) server; prepare; extract; aggregate ;;
  *) echo "usage: $0 [server|prepare|extract|aggregate|all]"; exit 1 ;;
esac
