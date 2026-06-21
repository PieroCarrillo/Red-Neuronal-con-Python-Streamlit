# Red Neuronal con Python + Streamlit

Proyecto de la **Opción 2** de la práctica de Redes Neuronales Artificiales. Este repositorio queda preparado para ejecutar una app de Streamlit que predice riesgo de diabetes usando una RNA y compara el enfoque con Gradient Boosted Trees.

## Contenido del repositorio

- `app.py`: aplicación Streamlit lista para desplegar.
- `01_entrenamiento.ipynb`: notebook de entrenamiento y comparación de modelos.
- `diabetes_data_es_GOLD.csv`: dataset para que la app pueda entrenar y ejecutarse en Streamlit Cloud.
- `requirements.txt`: dependencias necesarias.
- `URL_Streamlit.txt`: espacio para pegar la URL pública luego del deploy.
- `resultados/`: tablas CSV con métricas, correlación, importancia de variables y desafíos.
- `README_INSTRUCCIONES.txt`: pasos de ejecución y publicación.

## Resultados base del informe original

| Modelo | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| RNA MLP [32,16] | 0.9423 | 0.9516 | 0.9953 |
| Gradient Boosted Trees | 0.9904 | 0.9921 | 1.0000 |

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar en Streamlit Cloud

1. Entrar a Streamlit Cloud.
2. Crear una nueva app.
3. Seleccionar este repositorio.
4. Elegir `app.py` como archivo principal.
5. Presionar **Deploy**.
6. Copiar la URL pública en `URL_Streamlit.txt` y en el informe.

> Nota: los archivos binarios grandes, capturas e informe PDF completo están en el ZIP generado fuera del repositorio. Este repositorio contiene la versión necesaria para ejecutar y desplegar la app.
