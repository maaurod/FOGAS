#!/bin/bash
# Submit FOGAS Grid Search as a batch job on a GPU node
# Usage: ./submit_grid_search_job.sh [time] [gpu_count]

TIME=${1:-"12:00:00"}      # Default: 24 hours
GPU_COUNT=${2:-1}          # Default: 1 GPU
JOB_NAME="fogas-gridsearch"
LOG_DIR="/shared/home/mauro.diaz/logs/fogas"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}_%j.log"

echo "📤 Submitting FOGAS Grid Search job to GPU node..."

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --time=$TIME
#SBATCH --gres=gpu:$GPU_COUNT
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=$LOG_FILE
#SBATCH --error=$LOG_FILE

cd /shared/home/mauro.diaz/work/FOGAS
source venv/bin/activate

echo "✅ Job started at: \$(date)"
echo "🖥️  Running on node: \$(hostname)"
echo "🎮 GPUs available:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Run the grid search script
python3 /shared/home/mauro.diaz/work/FOGAS/testing_vectorized/grid_search_sbatch.py

echo ""
echo "✅ Job finished at: \$(date)"
EOF

echo ""
echo "✅ Job submitted! Monitor it with: squeue -u \$(whoami)"
echo "📋 Check the log with: tail -f \$(ls -t $LOG_DIR/${JOB_NAME}_*.log | head -n 1)"
echo "🛑 To cancel: scancel -n $JOB_NAME -u \$(whoami)"
