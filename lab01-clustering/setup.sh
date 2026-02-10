#!/bin/bash

# Script de setup para Lab 01 - Clustering
# Uso: ./setup.sh

echo "🚀 Configurando Lab 01 - Clustering"
echo ""

# Verificar que Python está instalado
if ! command -v python &> /dev/null; then
    echo "❌ Python no está instalado. Por favor instala Python 3.8+ primero."
    exit 1
fi

echo "✓ Python encontrado: $(python --version)"
echo ""

# Crear entorno virtual
echo "📦 Creando entorno virtual..."
python -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Error al crear el entorno virtual"
    exit 1
fi

echo "✓ Entorno virtual creado"
echo ""

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Error al activar el entorno virtual"
    exit 1
fi

echo "✓ Entorno virtual activado"
echo ""

# Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip --quiet

if [ $? -ne 0 ]; then
    echo "❌ Error al actualizar pip"
    exit 1
fi

echo "✓ pip actualizado"
echo ""

# Instalar dependencias
echo "📥 Instalando dependencias..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Error al instalar dependencias"
    exit 1
fi

echo "✓ Dependencias instaladas"
echo ""

# Verificar instalación
echo "🔍 Verificando instalación..."
python -c "import numpy; import pandas; import matplotlib; import seaborn; import sklearn; import jupyter"

if [ $? -ne 0 ]; then
    echo "❌ Error: Algunas librerías no se instalaron correctamente"
    exit 1
fi

echo "✓ Todas las librerías instaladas correctamente"
echo ""

echo "✅ ¡Setup completado exitosamente!"
echo ""
echo "📝 Próximos pasos:"
echo "   1. Activa el entorno virtual: source venv/bin/activate"
echo "   2. Inicia Jupyter: jupyter notebook"
echo "   3. Abre el notebook en notebooks/"
echo ""
echo "💡 Para desactivar el entorno: deactivate"
