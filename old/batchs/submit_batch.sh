#!/bin/bash
# Submit FOGAS RBF Grid Search as a batch job on a GPU node
# Usage: ./submit_batch.sh [time] [gpu_count] [partition]

TIME=${1:-"60:00:00"}      # Default: 48 hours
GPU_COUNT=${2:-1}          # Default: 1 GPU
PARTITION=${3:-"frida"}    # Default: frida partition
JOB_NAME="grid_search_50grid"
LOG_DIR="/shared/home/mauro.diaz/logs/fogas"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_NAME}_%j.log"

echo "📤 Submitting FOGAS RBF Grid Search job to $PARTITION partition ($GPU_COUNT GPU)..."

sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --time=$TIME
#SBATCH --partition=$PARTITION
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

# Run the RBF grid search script
python3 /shared/home/mauro.diaz/work/FOGAS/grid_search_fogas.py
echo ""
echo "✅ Job finished at: \$(date)"
SBATCH_EOF

echo ""
echo "✅ Job submitted! Monitor it with: squeue -u \$(whoami)"
echo "📋 Check the log with: tail -f \$(ls -t $LOG_DIR/${JOB_NAME}_*.log | head -n 1)"
echo "🛑 To cancel: scancel -n $JOB_NAME -u \$(whoami)"
