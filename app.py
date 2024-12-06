import streamlit as st
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

# Inicializar la conversión de pandas a R y viceversa
pandas2ri.activate()

# Cargar el modelo RDS
#@st.cache_resource
def cargar_modelo():
    with open('model.rds', 'rb') as file:
        modelo_r = robjects.r['readRDS'](file)
    return modelo_r

# Realizar predicciones usando el modelo cargado
def predecir_logP(modelo, input_df):
    r_df = pandas2ri.py2rpy(input_df)  # Convertir DataFrame de pandas a R
    predicciones = modelo.rx2('predict')(r_df)
    predicciones_py = pandas2ri.rpy2py(predicciones)  # Convertir de nuevo a pandas
    return predicciones_py

# Configurar la interfaz de Streamlit
st.title("Predicción de logP usando un modelo RDS")

# Subir archivo CSV
archivo_csv = st.file_uploader("Sube el archivo CSV con las moléculas", type=["csv"])

if archivo_csv is not None:
    # Leer archivo CSV
    input_data = pd.read_csv(archivo_csv)
    st.write("Datos cargados:")
    st.dataframe(input_data)
    
    # Cargar el modelo
    modelo_rds = cargar_modelo()
    
    # Predecir logP
    if st.button("Predecir logP"):
        predicciones = predecir_logP(modelo_rds, input_data)
        st.write("Predicciones de logP:")
        st.dataframe(predicciones)
