# Red Neuronal con Python + Streamlit

Proyecto de la **Opción 2** de la práctica de Redes Neuronales Artificiales. Incluye entrenamiento en Python, comparación entre RNA y Gradient Boosted Trees, aplicación Streamlit, capturas, resultados e informe.

## Contenido

- `01_entrenamiento.ipynb`: notebook ejecutado con carga de datos, correlación, entrenamiento, comparación de modelos, matrices de confusión y desafíos.
- `app.py`: aplicación Streamlit funcional.
- `diabetes_data_es_GOLD.csv`: dataset usado.
- `modelo_rna.keras`: modelo RNA guardado.
- `modelo_rna_sklearn.pkl`: respaldo del modelo para ejecutar la app sin TensorFlow.
- `modelo_gbt.pkl`: modelo Gradient Boosted Trees.
- `scaler.pkl`: normalizador usado antes de predecir.
- `requirements.txt`: dependencias.
- `informe_opcion2_RNA_Diabetes.pdf`: informe final.
- `capturas/`: capturas requeridas.
- `resultados/`: tablas de métricas, correlación, importancia de variables y desafíos.

## Resultados base

| Modelo | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| RNA MLP [32,16] | 0.9423 | 0.9516 | 0.9953 |
| Gradient Boosted Trees | 0.9904 | 0.9921 | 1.0000 |

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicación en Streamlit Cloud

1. Entrar a Streamlit Cloud.
2. Crear una nueva app.
3. Seleccionar este repositorio.
4. Elegir `app.py` como archivo principal.
5. Presionar **Deploy**.

> La URL pública debe copiarse en `URL_Streamlit.txt` y también colocarse en el informe antes de entregar.
