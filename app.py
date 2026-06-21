import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

st.set_page_config(page_title='Predictor Diabetes RNA', page_icon='🩺', layout='wide')

FEATURES = [
    'Edad', 'Genero', 'Poliuria', 'Polidipsia', 'Perdida_repentina_peso', 'Debilidad',
    'Polifagia', 'Candidiasis_genital', 'Vision_borrosa', 'Picazon', 'Irritabilidad',
    'Cicatrizacion_lenta', 'Paresia_parcial', 'Rigidez_muscular', 'Alopecia', 'Obesidad'
]

@st.cache_resource
def entrenar_modelos():
    df = pd.read_csv('diabetes_data_es_GOLD.csv')
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['TieneDiabetes'])
    X = df[FEATURES].values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    rna = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
                        max_iter=700, random_state=42, early_stopping=True,
                        validation_fraction=0.15)
    rna.fit(X_train_sc, y_train)

    gbt = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                     learning_rate=0.1, random_state=42)
    gbt.fit(X_train_sc, y_train)

    y_prob_rna = rna.predict_proba(X_test_sc)[:, 1]
    y_pred_rna = (y_prob_rna >= 0.5).astype(int)
    y_prob_gbt = gbt.predict_proba(X_test_sc)[:, 1]
    y_pred_gbt = gbt.predict(X_test_sc)

    metricas = pd.DataFrame({
        'Modelo': ['RNA MLP [32,16]', 'Gradient Boosted Trees'],
        'Accuracy': [accuracy_score(y_test, y_pred_rna), accuracy_score(y_test, y_pred_gbt)],
        'F1': [f1_score(y_test, y_pred_rna), f1_score(y_test, y_pred_gbt)],
        'AUC': [roc_auc_score(y_test, y_prob_rna), roc_auc_score(y_test, y_prob_gbt)]
    })
    return rna, gbt, scaler, metricas

modelo, modelo_gbt, scaler, metricas = entrenar_modelos()

st.title('🩺 Predictor de Diabetes — Red Neuronal Artificial')
st.caption('Aplicación Streamlit entrenada con el dataset diabetes_data_es_GOLD.csv')
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    edad = st.number_input('Edad', min_value=16, max_value=90, value=45, step=1)
    genero = st.selectbox('Género', ['Masculino (1)', 'Femenino (2)'])
    poliuria = st.checkbox('Poliuria')
    polidipsia = st.checkbox('Polidipsia')
    perdida = st.checkbox('Pérdida repentina de peso')
    debilidad = st.checkbox('Debilidad')
with col2:
    polifagia = st.checkbox('Polifagia')
    candidiasis = st.checkbox('Candidiasis genital')
    vision = st.checkbox('Visión borrosa')
    picazon = st.checkbox('Picazón')
    irritab = st.checkbox('Irritabilidad')
    cicatriz = st.checkbox('Cicatrización lenta')
with col3:
    paresia = st.checkbox('Paresia parcial')
    rigidez = st.checkbox('Rigidez muscular')
    alopecia = st.checkbox('Alopecia')
    obesidad = st.checkbox('Obesidad')

st.divider()

if st.button('🔍 Predecir Riesgo', use_container_width=True):
    g = 1 if 'Masculino' in genero else 2
    datos = np.array([[edad, g, int(poliuria), int(polidipsia), int(perdida), int(debilidad),
                       int(polifagia), int(candidiasis), int(vision), int(picazon), int(irritab),
                       int(cicatriz), int(paresia), int(rigidez), int(alopecia), int(obesidad)]])
    x = scaler.transform(datos)
    prob = float(modelo.predict_proba(x)[0][1])
    porcentaje = prob * 100
    if prob >= 0.5:
        st.error(f'⚠️ RIESGO ALTO de Diabetes — {porcentaje:.1f}%')
    else:
        st.success(f'✅ RIESGO BAJO de Diabetes — {porcentaje:.1f}%')
    st.progress(prob)
    st.caption('⚠️ Resultado orientativo. Consulte siempre a un profesional médico.')

with st.expander('Ver métricas base del proyecto'):
    st.dataframe(metricas.style.format({'Accuracy': '{:.4f}', 'F1': '{:.4f}', 'AUC': '{:.4f}'}), use_container_width=True)
    st.write('La app usa la RNA para la predicción principal y conserva GBT como modelo de comparación.')
