#!/bin/bash
# Script to start Jupyter Lab on a GPU node
# Usage: ./start_jupyter_gpu.sh [time] [gpu_count]
# Example: ./start_jupyter_gpu.sh 4:00:00 1

TIME=${1:-"12:00:00"}      # Default: 12 hours
GPU_COUNT=${2:-1}          # Default: 1 GPU
PORT=${3:-8888}            # Default port

echo "🚀 Starting Jupyter Lab on GPU node..."
echo "⏱️  Time: $TIME"
echo "🎮 GPUs: $GPU_COUNT"
echo "🔌 Port: $PORT"
echo "🚫 Excluding node: apl (slower L4 GPU)"
echo ""

# Start interactive session excluding the slower 'apl' node
# --exclude=apl \ after gres
srun \
     --time=$TIME \
     --gres=gpu:$GPU_COUNT \
     --mem=32G \
     --exclude=apl \
     --cpus-per-task=8 \
     --pty bash -c '
     cd /shared/home/mauro.diaz/work/FOGAS && \
     source venv/bin/activate && \
     export HOSTNAME=$(hostname) && \
     export USERNAME=$(whoami) && \
     echo "✅ Virtual environment activated" && \
     echo "🖥️  Running on node: $HOSTNAME" && \
     echo "🎮 GPUs available:" && \
     nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader && \
     echo "" && \
     echo "📓 Starting Jupyter Lab..." && \
     echo "🔗 To access, create an SSH tunnel from your local machine:" && \
     echo "   ssh -L '"$PORT"':$HOSTNAME:'"$PORT"' $USERNAME@login-frida" && \
     echo "" && \
     echo "Then open in browser: http://localhost:'"$PORT"'" && \
     echo "" && \
     jupyter lab --ip=0.0.0.0 --port='"$PORT"' --no-browser
     '
