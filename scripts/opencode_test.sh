#!/bin/bash
# Comprehensive test matching OpenCode's request pattern
AP="http://localhost:8000"
AK="wsh101007"

echo "=== Test 1: Basic chat (你好) ==="
curl -s -w '\nHTTP: %{http_code}\n' -X POST "$AP/v1/chat/completions" \
  -H "Authorization: Bearer $AK" \
  -H "Content-Type: application/json" \
  -d '{"model":"minicpm1","messages":[{"role":"user","content":"你好"}],"max_tokens":8,"temperature":0.1}'

echo ""
echo "=== Test 2: Streaming ==="
curl -s -w '\nHTTP: %{http_code}\n' -X POST "$AP/v1/chat/completions" \
  -H "Authorization: Bearer $AK" \
  -H "Content-Type: application/json" \
  -d '{"model":"minicpm1","messages":[{"role":"user","content":"你好"}],"max_tokens":4,"temperature":0.1,"stream":true}' | head -5

echo ""
echo "=== Test 3: Without model field ==="
curl -s -w '\nHTTP: %{http_code}\n' -X POST "$AP/v1/chat/completions" \
  -H "Authorization: Bearer $AK" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}'

echo ""
echo "=== Test 4: Without auth ==="
curl -s -w '\nHTTP: %{http_code}\n' -X POST "$AP/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"minicpm1","messages":[{"role":"user","content":"hi"}],"max_tokens":4}'
