#!/bin/bash
# llm-ascend310 — All-in-One Setup & Deploy Script
# Usage: ./deploy.sh [start|stop|restart|compile|status]

set -e

BASE="/root/llm-ascend310"
MODEL_PATH="/root/models/MiniCPM5-1B"
OM_DIR="$BASE/om_models"
LOG="/root/llm_server.log"
PIDFILE="/root/llm_server.pid"
PORT=${PORT:-8000}

export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/7.0.0/lib64:\
/usr/local/Ascend/ascend-toolkit/7.0.0/lib64/plugin/opskernel:\
/usr/local/Ascend/ascend-toolkit/7.0.0/lib64/plugin/nnengine:\
/usr/local/Ascend/driver/lib64:\
/usr/local/Ascend/driver/lib64/common:\
/usr/local/Ascend/driver/lib64/driver
export LLM_API_KEY=${LLM_API_KEY:?Must set LLM_API_KEY}
export LLM_MODEL_PATH=$MODEL_PATH
export LLM_MAX_CONTEXT=${LLM_MAX_CONTEXT:-131072}
export LLM_INSTANCES=${LLM_INSTANCES:-4}
export PYTHONPATH=$BASE:$PYTHONPATH

compile_ops() {
    echo "=== Compiling operators for MiniCPM5-1B ==="
    mkdir -p "$OM_DIR"
    apt-get install -y onnx 2>/dev/null || pip3 install onnx 2>&1 | tail -1
    cd "$BASE"
    python3 -m engine.compile --model "$MODEL_PATH" --output "$OM_DIR" --soc Ascend310
    echo "Done."
}

start() {
    if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        echo "Server already running (PID $(cat $PIDFILE))"
        exit 1
    fi

    # Compile ops if missing
    if [ ! -f "$OM_DIR/mm_1_1536_2048.om" ]; then
        compile_ops
    fi

    cd "$BASE"
    nohup python3 server/api.py > "$LOG" 2>&1 &
    PID=$!
    echo $PID > "$PIDFILE"
    echo "Started (PID $PID)"
    echo "Log: $LOG"
    echo "API: http://$(hostname -I | awk '{print $1}'):$PORT"
    sleep 10
    if kill -0 $PID 2>/dev/null; then
        echo "Server is running."
    else
        echo "Server crashed! Check log: tail -50 $LOG"
    fi
}

stop() {
    if [ -f "$PIDFILE" ]; then
        kill $(cat "$PIDFILE") 2>/dev/null || true
        rm -f "$PIDFILE"
    fi
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    pkill -f 'server/api.py' 2>/dev/null || true
    sleep 2
    echo "Stopped"
}

status() {
    if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        echo "Server running (PID $(cat $PIDFILE))"
        echo "Instances: $LLM_INSTANCES"
        npu-smi info 2>&1 | grep -E '^\|'
        echo "Recent log:"
        tail -5 "$LOG" 2>/dev/null
    else
        echo "Server not running"
    fi
}

setup() {
    echo "=== llm-ascend310 Setup ==="

    # Create directories
    mkdir -p "$BASE"/{engine,server,models,om_models,scripts,tests,docs}

    # Check CANN
    if [ -f /usr/local/Ascend/ascend-toolkit/7.0.0/compiler/version.info ]; then
        echo "✅ CANN 7.0.0"
    else
        echo "⚠️  CANN not found at expected path"
    fi

    # Check NPU
    npu-smi info 2>/dev/null && echo "✅ NPU" || echo "⚠️  No NPU detected"

    # Check model
    if [ -f "$MODEL_PATH/config.json" ]; then
        echo "✅ Model: $(python3 -c "import json;c=json.load(open('$MODEL_PATH/config.json'));print(c.get('_name_or_path','MiniCPM5-1B'))")"
    else
        echo "⚠️  Model not found at $MODEL_PATH"
    fi

    # Install deps
    pip3 install huggingface_hub onnx transformers tokenizers safetensors uvicorn fastapi pydantic 2>&1 | tail -1
    echo "✅ Python deps"

    echo "Setup complete. Run './deploy.sh compile' then './deploy.sh start'"
}

case "${1:-help}" in
    start)    start ;;
    stop)     stop ;;
    restart)  stop; sleep 2; start ;;
    status)   status ;;
    compile)  compile_ops ;;
    setup)    setup ;;
    *)        echo "Usage: $0 {start|stop|restart|status|compile|setup}" ;;
esac
