import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Membaca data
file_path = "C:\Streamlit\heart.csv" 
data = pd.read_csv('data/heart.csv')


# Menampilkan teks di aplikasi
st.title("Analisis Dataset Penyakit Jantung")
st.write("Ini adalah aplikasi analisis dataset penyakit jantung.")

# Menampilkan dataset
st.subheader("Data Awal")
st.write(data.head())

# Membuat distribusi target
st.subheader("Distribusi Target")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=data, x='target', palette='Set2', ax=ax)
st.pyplot(fig)

# Menampilkan heatmap korelasi
st.subheader("Heatmap Korelasi")
fig, ax = plt.subplots(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
st.pyplot(fig)
