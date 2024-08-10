import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.title("Dashboard Simples - Gráfico Interativo")

st.write("""
Este dashboard permite que você selecione a função que deseja visualizar no gráfico: 
( y = x^2 ) ou ( y = x^3 ).
Use a barra lateral para selecionar a função.
""")

function = st.sidebar.selectbox(
    "Selecione a função para plotar",
    ("y = x^2", "y = x^3")
)

x = np.linspace(-10, 10, 400)

if function == "y = x^2":
    y = x**2
else:
    y = x**3

df = pd.DataFrame({"x": x, "y": y})

fig = px.line(df, x="x", y="y", title=f"Gráfico da Função {function}")

st.plotly_chart(fig)
