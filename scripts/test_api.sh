#!/bin/bash
# Full API compatibility test for Codex
PASS=0; FAIL=0

test() {
    local desc="$1"; shift
    local expect="$1"; shift
    local result=$(curl -s -o /dev/null -w '%{http_code}' "$@" 2>/dev/null)
    if [ "$result" = "$expect" ]; then
        echo "  ✅ $desc → $result"
        PASS=$((PASS+1))
    else
        echo "  ❌ $desc → $result (expected $expect)"
        FAIL=$((FAIL+1))
    fi
}

echo "=== Full API Test Suite ==="

# Health
test "GET /health" 200 http://localhost:8000/health
test "GET /" 200 http://localhost:8000/

# Models
test "GET /v1/models (with auth)" 200 -H "Authorization: Bearer wsh101007" http://localhost:8000/v1/models
test "GET /v1/models (no auth)" 401 http://localhost:8000/v1/models

# Chat
test "POST chat (minicpm1)" 200 -H "Authorization: Bearer wsh101007" -H "Content-Type: application/json" -d '{"model":"minicpm1","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' http://localhost:8000/v1/chat/completions
test "POST chat (no auth)" 401 -H "Content-Type: application/json" -d '{"model":"minicpm1","messages":[]}' http://localhost:8000/v1/chat/completions
test "POST chat (MiniCPM1)" 200 -H "Authorization: Bearer wsh101007" -H "Content-Type: application/json" -d '{"model":"MiniCPM1","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' http://localhost:8000/v1/chat/completions

# CORS
test "OPTIONS preflight" 200 -X OPTIONS -H "Origin: http://test.com" -H "Access-Control-Request-Method: POST" http://localhost:8000/v1/chat/completions

# Completions (text)
test "POST completions" 200 -H "Authorization: Bearer wsh101007" -H "Content-Type: application/json" -d '{"model":"minicpm1","prompt":"Hello","max_tokens":4}' http://localhost:8000/v1/completions

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
