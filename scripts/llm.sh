#!/bin/bash
# llm - MiniCPM5-1B @ Ascend310 CLI Management Tool
# Usage: llm <command> [options]

BASE="/root/llm-ascend310"
LOG="/root/llm_server.log"
PIDFILE="/root/llm_server.pid"
MODEL="/root/models/MiniCPM5-1B"
PORT=8000

# ── Environment ──
LLM_ENV="LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/7.0.0/lib64:\
/usr/local/Ascend/ascend-toolkit/7.0.0/lib64/plugin/opskernel:\
/usr/local/Ascend/ascend-toolkit/7.0.0/lib64/plugin/nnengine:\
/usr/local/Ascend/driver/lib64:\
/usr/local/Ascend/driver/lib64/common:\
/usr/local/Ascend/driver/lib64/driver"
PY_ENV="PYTHONPATH=$BASE"

# ── Helpers ──
get_pid() { cat "$PIDFILE" 2>/dev/null; }
is_running() { [ -f "$PIDFILE" ] && kill -0 $(get_pid) 2>/dev/null; }

check_temp() {
    local line=$(npu-smi info 2>/dev/null | sed -n '4p')
    local temp=$(echo "$line" | awk '{print $7}' | sed 's/°C//')
    if [ -z "$temp" ] || ! [ "$temp" -eq "$temp" ] 2>/dev/null; then
        echo "Unable to read NPU temperature"
        return 1
    fi
    echo "NPU: ${temp}C"
    if [ "$temp" -ge 95 ]; then return 1; fi
    return 0
}

ensure_cool() {
    while ! check_temp; do
        echo "Waiting for NPU to cool (check again in 30s)..."
        sleep 30
    done
}

# ── Commands ──
cmd_start() {
    if is_running; then
        echo "✅ Server already running (PID $(get_pid))"
        return 0
    fi
    ensure_cool
    cd "$BASE"
    nohup env $LLM_ENV $PY_ENV LLM_API_KEY="${LLM_API_KEY:-llm101007}" \
        python3 server/api.py > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo -n "Starting (PID $!)..."
    for i in $(seq 1 30); do
        sleep 1
        if curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
            echo " ✅"
            echo "  $(curl -s http://localhost:$PORT/health)"
            echo "  API: http://$(hostname -I | awk '{print $1}'):$PORT"
            return 0
        fi
        echo -n "."
    done
    echo " ❌ Timeout"
    tail -10 "$LOG"
    return 1
}

cmd_stop() {
    if ! is_running; then
        echo "ℹ️  Server not running"
        return 0
    fi
    echo -n "Stopping (PID $(get_pid))..."
    kill $(get_pid) 2>/dev/null
    rm -f "$PIDFILE"
    fuser -k "${PORT}/tcp" 2>/dev/null
    for i in $(seq 1 10); do
        if ! curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
            echo " ✅"
            return 0
        fi
        sleep 1
    done
    kill -9 $(get_pid) 2>/dev/null
    echo " 💀 Force killed"
}

cmd_restart() { cmd_stop; sleep 2; cmd_start; }

cmd_status() {
    echo "=== llm-ascend310 Status ==="
    if is_running; then
        local pid=$(get_pid)
        local uptime=$(ps -o etime= -p $pid 2>/dev/null | tr -d ' ')
        echo "📊 Server:  ✅ Running (PID $pid, up ${uptime:-?})"
        echo "   Port:    $PORT"
        echo "   Memory:  $(ps -o rss= -p $pid 2>/dev/null | awk '{printf "%.0fMB", $1/1024}')"
    else
        echo "📊 Server:  ❌ Stopped"
    fi
    echo ""
    npu-smi info 2>/dev/null | grep -E '^\|' | grep -v '^+\|No run\|^$' | head -8
    echo ""
    echo "📁 Model:    $MODEL"
    echo "   Disk:     $(df -h / | tail -1 | awk '{print $3 " / " $2 " (" $5 ")"}')"
    echo "   Log:      $LOG (lines: $(wc -l < "$LOG" 2>/dev/null || echo 0))"
}

cmd_logs() {
    local lines=${1:-30}
    if [ "$1" = "-f" ] || [ "$1" = "--follow" ]; then
        tail -f "$LOG"
    else
        tail -${lines} "$LOG"
    fi
}

cmd_config() {
    echo "=== llm-ascend310 Configuration ==="
    echo "  Server:   http://$(hostname -I | awk '{print $1}'):$PORT"
    echo "  Model:    $MODEL"
    echo "  API Key:  ${LLM_API_KEY:-llm101007}"
    echo ""
    echo "  Environment Variables:"
    echo "    LLM_API_KEY       Auth key (default: llm101007)"
    echo "    LLM_MAX_CONTEXT   Max context (default: 32768)"
    echo "    LLM_INSTANCES     Instances (default: 4, multi-chip)"
    echo "    PORT              Server port (default: 8000)"
    echo ""
    echo "  Files:"
    echo "    Engine:   $BASE/"
    echo "    Log:      $LOG"
    echo "    PID:      $PIDFILE"
}

cmd_bench() {
    if ! is_running; then
        echo "❌ Server not running. Start it first: llm start"
        return 1
    fi
    echo "Running benchmark..."
    python3 "$BASE/tests/benchmark.py" "$@"
}

cmd_test() {
    if ! is_running; then
        echo "❌ Server not running"
        return 1
    fi
    python3 "$BASE/tests/test_api.sh"
}

# ── Main ──
case "${1:-help}" in
    start|stop|restart)   cmd_${1} ;;
    status)               cmd_status ;;
    logs|log)             shift; cmd_logs "$@" ;;
    config|conf)          cmd_config ;;
    bench|benchmark)      shift; cmd_bench "$@" ;;
    test)                 cmd_test ;;
    help|--help|-h)
        echo "Usage: llm <command>"
        echo ""
        echo "Commands:"
        echo "  start       Start server"
        echo "  stop        Stop server"
        echo "  restart     Restart server"
        echo "  status      Server status + NPU health"
        echo "  logs [-f]   View logs (tail)"
        echo "  config      Show configuration"
        echo "  bench       Run benchmark"
        echo "  test        Run API tests"
        echo ""
        echo "Env:"
        echo "  LLM_API_KEY=your_key  llm start"
        ;;
    *) echo "Unknown: $1"; exit 1 ;;
esac
