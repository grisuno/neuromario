#!/bin/bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt


# Nombre del entorno
ENV_NAME="trimario-env"

echo "🔹 Creando entorno Conda: $ENV_NAME"

# Crear entorno con Python 3.9 (versión estable para compatibilidad con gym y nes-py)
conda create -n $ENV_NAME python=3.9 -y

# Activar entorno
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "🔹 Instalando dependencias base desde conda-forge"

# Instalar PyTorch CPU (compatible con tu Config.DEVICE = "cpu")
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Instalar OpenCV, NumPy, Matplotlib y demás desde conda-forge (más estable)
conda install -c conda-forge numpy matplotlib opencv tqdm scikit-learn -y

echo "🔹 Instalando dependencias específicas de RL y entornos"

# Instalar gym 0.21 (última versión compatible con wrappers antiguos)
pip install gym==0.21.0

# Instalar gym-super-mario-bros y nes-py (versión que funciona con gym 0.21)
pip install nes-py==8.2.1
pip install gym-super-mario-bros==7.4.0

echo "🔹 Instalando UMAP para visualización 3D (opcional pero recomendado)"

# UMAP para la visualización en 3D
pip install umap-learn

echo "🔹 Instalando paquetes adicionales de compatibilidad"

# Asegurar compatibilidad de cv2 y otros
pip install Pillow

echo "✅ Entorno '$ENV_NAME' configurado exitosamente."

echo "📌 Para usarlo, ejecuta:"
echo "    conda activate $ENV_NAME"
echo "    python visualizacion_foco_real.py"

# Opcional: desactivar entorno
conda deactivate
