import streamlit as st
import pandas as pd
import pickle

# Cargar modelo
filename = 'modelo-reg-tree-knn-nn.pkl'
with open(filename, 'rb') as f:
    model_tree, model_KNN, model_NN, variables, min_max_scaler = pickle.load(f)

# Obtener columnas dummy
videojuegos_dummies = [v for v in variables if v.startswith("videojuego_")]
plataformas_dummies = [v for v in variables if v.startswith("Plataforma_")]

# Diccionarios limpios para mostrar al usuario
videojuego_map = {v.replace("videojuego_", "").replace("_", " "): v for v in videojuegos_dummies}
plataforma_map = {v.replace("Plataforma_", "").replace("_", " "): v for v in plataformas_dummies}

# Imagen personalizada
st.image(
    "https://cdn.pixabay.com/photo/2017/01/31/13/14/controller-2020207_960_720.png",
    caption="Bienvenido a la predicción de presupuesto gamer",
    use_container_width=True
)

# Encabezado
st.markdown("## 🧠 Predice cuánto gastará tu cliente gamer")
st.markdown("Completa los datos y obtén un presupuesto estimado.")

# Diseño con columnas
col1, col2 = st.columns(2)
with col1:
    edad = st.slider("Edad del cliente", 14, 52, 25)
    sexo = st.radio("Sexo", ["Hombre", "Mujer"])
    consumidor = st.checkbox("Consumidor habitual", value=True)
with col2:
    videojuego = st.selectbox("🎮 Videojuego preferido", list(videojuego_map.keys()))
    plataforma = st.selectbox("🕹️ Plataforma preferida", list(plataforma_map.keys()))

# Crear input
input_data = pd.DataFrame(columns=variables)
input_data.loc[0] = 0
input_data["Edad"] = min_max_scaler.transform([[edad]])[0][0]
input_data[videojuego_map[videojuego]] = 1
input_data[plataforma_map[plataforma]] = 1
if sexo == "Mujer":
    input_data["Sexo_Mujer"] = 1
if consumidor:
    input_data["Consumidor_habitual_True"] = 1

# Resultado
if st.button("🔍 Estimar presupuesto"):
    pred = model_NN.predict(input_data)[0]
    st.success(f"💸 Presupuesto estimado: ${pred:,.2f}")
