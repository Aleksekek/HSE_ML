import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from models.data_preprocessor import CarDataPreprocessor
from sklearn.linear_model import ElasticNet

st.title("MEGA DATA EXPLORER")

st.header("EDA по тренировочным данным 📊")

PLOTS_DIR = "saved_plots"
try:
    df = pd.read_csv(r"data/eda_data.csv")

    st.subheader("Описание данных (describe)")
    st.dataframe(df.describe().T)
except:
    st.text("Не обнаружены данные по пути data/eda_data.csv, блок describe пропущен")

st.subheader("Тепловая карта корреляций Phik")
phik_matrix_path = os.path.join(PLOTS_DIR, "phik_matrix.png")
st.image(phik_matrix_path)

st.subheader("Распределение цен на автомобили")
prices = os.path.join(PLOTS_DIR, "prices.png")
st.image(prices)

st.subheader("Распределение цен с логарифмированием")
prices_log = os.path.join(PLOTS_DIR, "prices_log.png")
st.image(prices_log)

st.subheader("Выбросы в данных")
outliers = os.path.join(PLOTS_DIR, "outliers.png")
st.image(outliers)


st.header("Прогнозирование цен на загруженных данных 💸")


@st.cache_resource
def load_preprocessor():
    with open("models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return preprocessor


@st.cache_resource
def load_model():
    with open("models/elasticnet_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)


uploaded_file = st.file_uploader("Загрузите CSV", type=["csv"])
if uploaded_file:
    user_df = load_data(uploaded_file)

    assert (
        user_df.shape[1] == 12
    ), "Ошибка при загрузке данных: неверное количество столбцов! (Должно быть 12)"

    columns = [
        "name",
        "year",
        "km_driven",
        "fuel",
        "seller_type",
        "transmission",
        "owner",
        "mileage",
        "engine",
        "max_power",
        "torque",
        "seats",
    ]
    assert sorted(user_df.columns.to_list()) == sorted(
        columns
    ), f"Ошибка при загрузке данных: неверные столбцы! (Должны быть: {columns})"

    user_df = user_df[columns]  # Для восстановления порядка колонок при необходимости

    st.markdown("**Загрузка данных прошла успешно!**")
else:
    user_df = None

if isinstance(user_df, pd.DataFrame):
    processor = load_preprocessor()
    user_df_transformed = processor.transform(user_df)

    assert user_df_transformed.shape[1] == 98, "Ошибка при предобработке данных"

    st.markdown("**Предобработка данных прошла успешно!**")

    model = load_model()

    predictions = np.exp(model.predict(np.array(user_df_transformed)))
    predictions = pd.DataFrame(predictions, columns=["selling_price"])

    st.markdown("**Прогнозирование выполнено успешно!**")
    st.dataframe(predictions, height=350)

    # Использовал для этого блока deepseek, так как не смог сам быстро поправить ошибку с неверным типом данных для загрузки
    if not predictions.empty:
        csv_data = predictions.to_csv(index=False)

        st.download_button(
            label="Скачать полученные предсказания",
            data=csv_data,
            file_name="selling_price.csv",
            mime="text/csv",
        )
    else:
        st.warning("Нет данных для скачивания")


st.header("Веса обученной модели 📶")
elasticnet_coefs = os.path.join(PLOTS_DIR, "elasticnet_coefs.png")
st.image(elasticnet_coefs)
