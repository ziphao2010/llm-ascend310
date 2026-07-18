#!/bin/bash
echo "=== Stress Test: 10 sequential requests ==="
for i in $(seq 1 10); do
    result=$(curl -s -o /dev/null -w '%{http_code}' \
        -X POST http://localhost:8000/v1/chat/completions \
        -H 'Authorization: Bearer wsh101007' \
        -H 'Content-Type: application/json' \
        -d '{"model":"minicpm1","messages":[{"role":"user","content":"hi"}],"max_tokens":4}')
    echo "  Request $i: $result"
    sleep 1
done
echo "=== Memory ==="
npu-smi info 2>&1 | grep -E '^\|.*0.*\|.*$'
echo "Done"
