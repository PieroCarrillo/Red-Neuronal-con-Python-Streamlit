PRACTICA SEMANA 03 - OPCION 2: PYTHON + STREAMLIT
Proyecto: Predictor de Diabetes con Red Neuronal Artificial

Contenido principal:
- 01_entrenamiento.ipynb: notebook con carga de datos, correlación, entrenamiento, comparación RNA vs GBT, matrices de confusión y desafíos.
- app.py: aplicación Streamlit funcional.
- diabetes_data_es_GOLD.csv: dataset para ejecución de la app en Streamlit.
- requirements.txt: dependencias para ejecutar la app.
- resultados/: tablas CSV con métricas, correlaciones, feature importances y desafíos.

Resultados base del informe original:
RNA MLP [32,16] -> Accuracy=0.9423, F1=0.9516, AUC=0.9953
Gradient Boosted Trees -> Accuracy=0.9904, F1=0.9921, AUC=1.0000

Cómo ejecutar localmente:
1. Instalar dependencias: pip install -r requirements.txt
2. Ejecutar app: streamlit run app.py
3. Abrir la URL local que aparece en la terminal.

Cómo publicar:
1. Entrar a Streamlit Cloud.
2. New App -> elegir este repositorio -> app.py -> Deploy.
3. Copiar la URL pública en URL_Streamlit.txt y en el informe antes de entregar.

Nota:
Los archivos binarios y el PDF completo quedaron en el ZIP generado fuera del repositorio. El repositorio queda listo para ejecutar y desplegar la app Streamlit.