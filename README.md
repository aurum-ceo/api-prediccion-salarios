# Proyecto 09: API de Predicción de Salarios con Machine Learning

[![CI](https://github.com/angel-alvarado/proyecto-09/actions/workflows/ci.yml/badge.svg)](https://github.com/angel-alvarado/proyecto-09/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![codecov](https://codecov.io/gh/angel-alvarado/proyecto-09/branch/main/graph/badge.svg)](https://codecov.io/gh/angel-alvarado/proyecto-09)

## Descripción

Una API RESTful construida con FastAPI que sirve un modelo de Machine Learning para predecir salarios en ciencia de datos basado en características como puesto, experiencia, tipo de empleo, modalidad remota y tamaño de empresa.

## Características

- 🚀 **FastAPI**: Alto rendimiento, documentación automática (Swagger UI, ReDoc)
- 🧠 **Machine Learning**: Modelo de Random Forest entrenado con scikit-learn
- 🐳 **Docker**: Imagen lista para producción con multi-stage build
- 🔬 **Testing**: Cobertura de pruebas con pytest
- 🛡️ **Seguridad**: Escaneo de dependencias con Safety y Bandit
- 📊 **Monitoreo**: Endpoints de health y métricas
- 📚 **Documentación**: Documentación generada automáticamente y sitio con MkDocs
- ⚙️ **CI/CD**: Pipeline de GitHub Actions para lint, test, build y deploy
- 🎯 **Calidad**: Pre-commit hooks, type checking (mypy), formateo (black, isort)
- 📈 **Versionado**: Semantic versioning con changelog

## Empezando

### Prerrequisitos

- Python 3.11+
- Docker (opcional, para contenedorización)
- Git

### Instalación local

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/proyecto-09.git
cd proyecto-09

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Para desarrollo

# Instalar pre-commit hooks
pre-commit install
```

### Ejecutar la API

```bash
# Modo desarrollo (recarga automática)
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# O usar el script de ayuda
make run
```

La API estará disponible en http://localhost:8000
Documentación interactiva: http://localhost:8000/docs
```

### Ejecutar con Docker

```bash
# Construir la imagen
docker build -t proyecto-09:latest .

# Ejecutar el contenedor
docker run -p 8000:8000 proyecto-09:latest
```

### Ejecutar pruebas

```bash
# Ejecutar suite de pruebas
pytest

# Ejecutar con cobertura
pytest --cov=src --cov-report=term-missing

# O usar make
make test
```

## Estructura del Proyecto

```
proyecto-09/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── src/
│   ├── main.py                 # Entrypoint de FastAPI
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # Endpoints de la API
│   └── ml/
│       ├── __init__.py
│       ├── model.py            # Definición y carga del modelo
│       └── train.py            # Script para entrenar el modelo
├── tests/
│   ├── __init__.py
│   │   └── test_api.py         # Pruebas de la API
│   └── test_ml.py              # Pruebas del modelo
├── docs/
│   ├── index.md                # Página principal de documentación
│   └── mkdocs.yml              # Configuración de MkDocs
├── Dockerfile                  # Definición de imagen Docker
├── docker-compose.yml          # Servicios auxiliares (ej. base de datos)
├── requirements.txt            # Dependencias de producción
├── requirements-dev.txt        # Dependencias de desarrollo y testing
├── .pre-commit-config.yaml     # Hooks de pre-commit
├── .gitignore                  # Archivos a ignorar en Git
├── Makefile                    # Atajos para tareas comunes
├── LICENSE                     # Licencia MIT
└── README.md                   # Este archivo
```

## Uso de la API

### Endpoints disponibles

- `GET /` - Información básica de la API
- `GET /health` - Health check
- `POST /predict` - Realizar una predicción de salario
- `GET /model/info` - Información sobre el modelo entrenado
- `GET /docs` - Documentación interactiva Swagger UI
- `GET /redoc` - Documentación ReDoc

### Ejemplo de predicción

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "job_title": "Data Scientist",
    "experience_level": "SE",
    "employment_type": "FT",
    "remote_ratio": 100,
    "company_size": "L"
  }'
```

Respuesta:
```json
{
  "predicted_salary_usd": 125000.0,
  "confidence_interval": {
    "lower": 110000.0,
    "upper": 140000.0
  },
  "model_version": "1.0.0",
  "timestamp": "2026-09-21T12:00:00Z"
}
```

## Entrenamiento del modelo

Para re-entrenar el modelo con nuevos datos:

```bash
python -m src.ml.train
```

Esto leerá `data/salaries.csv` (si existe) y generará un nuevo modelo en `models/salary_model.joblib`.

## Despliegue

El pipeline de CI/CD construye automáticamente la imagen Docker y la sube al GitHub Container Registry (GHCR) cuando se crea un tag de versión (v*.*.*).

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Créditos

Creado con ❤️ usando FastAPI, scikit-learn y las mejores prácticas de desarrollo de software.
