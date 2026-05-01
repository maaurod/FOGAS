# FOGAS Examples

This directory contains example notebooks demonstrating the FOGAS algorithm.

## Vectorized Examples (Recommended)

Located in `testing_vectorized/`:

### 1. **2State.ipynb** - Two-State MDP
Simple example with a 2-state MDP. Good starting point for understanding the basics.

**Features:**
- Basic FOGAS solver usage
- Oracle solver comparison
- Hyperparameter optimization
- Performance evaluation

### 2. **3grid.ipynb** - 3x3 Gridworld Navigation
Standard gridworld navigation problem.

**Features:**
- Larger state/action space
- Multiple solver configurations
- Data collection from environment
- Comparative analysis

### 3. **3grid_wall.ipynb** - 3x3 Gridworld with Obstacles
Gridworld with obstacles and terminal states.

**Features:**
- Environment with absorbing states
- Dataset collection demonstration
- Policy visualization

### 4. **10grid_wall.ipynb** - 10x10 Gridworld
Larger-scale gridworld problem.

## Standard Examples

Located in `testing/`:

Similar examples using the non-vectorized solver implementation.

## Running in Google Colab

1. Upload a notebook to Colab or open directly from GitHub
2. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
3. Run the cells in order

All notebooks use relative paths and will work seamlessly in Colab.
