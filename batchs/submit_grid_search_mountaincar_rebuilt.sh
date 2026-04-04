#!/bin/bash
# Submit MountainCar rebuilt grid search as a batch job on a GPU node
# Usage: bash submit_grid_search_mountaincar_rebuilt.sh [time] [gpu_count]

TIME=${1:-"20:00:00"}
GPU_COUNT=${2:-1}
JOB_NAME="mc_rebuilt"

LOG_DIR="/shared/home/mauro.diaz/logs/fogas"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}_%j.log"

echo "📤 Submitting MountainCar rebuilt grid search job to the frida partition..."
echo "   GPUs       : $GPU_COUNT × A100_80GB"
echo "   Log        : $LOG_FILE"
echo ""

sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --time=$TIME
#SBATCH --partition=frida
#SBATCH --gres=gpu:A100_80GB:$GPU_COUNT
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

python3 /shared/home/mauro.diaz/work/FOGAS/testing_vectorized/utils/grid_search_mountaincar_rebuilt.py

echo ""
echo "✅ Job finished at: \$(date)"
SBATCH_EOF

echo ""
echo "✅ Job submitted! Monitor it with: squeue -u \$(whoami)"
echo "📋 Check the log with: tail -f \$(ls -t $LOG_DIR/${JOB_NAME}_*.log | head -n 1)"
echo "🛑 To cancel: scancel -n $JOB_NAME -u \$(whoami)"
