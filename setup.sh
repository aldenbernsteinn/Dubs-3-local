#!/bin/bash
# QWEN-M5 Setup — raises GPU memory limit and activates venv

echo "Raising GPU wired memory limit to 40GB..."
sudo sysctl iogpu.wired_limit_mb=40960

echo "Activating MLX environment..."
source ~/mlx-env/bin/activate

echo "MLX version:"
python3 -c "import mlx; print(mlx.__version__)"

echo "Setup complete."
