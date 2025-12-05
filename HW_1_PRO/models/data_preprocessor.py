import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CarDataPreprocessor:
    def __init__(self, brand_country_mapping, numerical_features, interaction_features):
        self.brand_country_mapping = brand_country_mapping
        self.numerical_features = numerical_features
        self.interaction_features = interaction_features

        self.medians = {}
        self.train_means = {}
        self.train_stds = {}
        self.train_mins = {}
        self.ohe = None
        self.scaler = None
        self.feature_names = None
        self.final_feature_names_ = None

    def parse_torque(self, torque_str):
        if pd.isna(torque_str) or str(torque_str).strip() in ["", "torque"]:
            return np.nan, np.nan

        torque_str = str(torque_str).strip()
        numbers = re.findall(r"(\d+\.?\d*)", torque_str.replace(",", ""))

        torque_val = np.nan
        rpm_val = np.nan

        if len(numbers) >= 1:
            torque_val = float(numbers[0])
            if "kgm" in torque_str.lower() or "kg" in torque_str.lower():
                torque_val = torque_val * 9.80665

        if len(numbers) >= 2:
            rpm_val = float(numbers[1])

        return torque_val, rpm_val

    def fix_column_types(self, df):
        """Исправление типов колонок"""
        fixed_df = df.copy()

        # Mileage
        fixed_df["mileage"] = fixed_df["mileage"].apply(
            lambda x: (
                float(str(x).split()[0])
                if str(x).endswith("kmpl") and x.strip() != "kmpl"
                else (
                    float(str(x).split()[0]) * 1.40
                    if str(x).endswith("km/kg") and x.strip() != "km/kg"
                    else np.nan
                )
            )
        )

        # Engine
        fixed_df["engine"] = fixed_df["engine"].apply(
            lambda x: (
                float(str(x).split()[0])
                if str(x).endswith("CC") and x.strip() != "CC"
                else np.nan
            )
        )

        # Max Power
        fixed_df["max_power"] = fixed_df["max_power"].apply(
            lambda x: (
                float(str(x).split()[0])
                if str(x).endswith("bhp") and x.strip() != "bhp"
                else np.nan
            )
        )

        # Torque
        torque_results = fixed_df["torque"].apply(self.parse_torque)
        fixed_df["torque"] = torque_results.apply(lambda x: x[0])
        fixed_df["max_torque_rpm"] = torque_results.apply(lambda x: x[1])

        return fixed_df

    def extract_brand_and_country(self, df):
        """Извлечение бренда и страны"""
        df_fe = df.copy()
        df_fe["name"] = df_fe["name"].apply(lambda x: str(x).split()[0])
        df_fe["country"] = df_fe["name"].map(self.brand_country_mapping)
        return df_fe

    def create_polynomial_features(self, df):
        """Создание полиномиальных features"""
        df_fe = df.copy()

        # Квадраты
        for feature in self.numerical_features:
            if feature in df_fe.columns:
                df_fe[f"{feature}_squared"] = df_fe[feature] ** 2

        # Нормализация + sin
        for feature in self.numerical_features:
            if (
                feature in df_fe.columns
                and hasattr(self, "train_means")
                and feature in self.train_means
            ):
                if self.train_stds[feature] != 0:
                    normalized = (
                        df_fe[feature] - self.train_means[feature]
                    ) / self.train_stds[feature]
                    df_fe[f"{feature}_sin"] = np.sin(normalized)
                else:
                    df_fe[f"{feature}_sin"] = 0

        return df_fe

    def create_log_features(self, df):
        """Создание логарифмических features"""
        df_fe = df.copy()

        for feature in self.numerical_features:
            if (
                feature in df_fe.columns
                and hasattr(self, "train_mins")
                and feature in self.train_mins
            ):
                min_val = self.train_mins[feature]
                constant = abs(min_val) + 1 if min_val <= 0 else 0
                # Защита от логарифма неположительных чисел
                values = df_fe[feature] + constant
                values = np.where(values <= 0, 1e-10, values)
                df_fe[f"{feature}_log"] = np.log(values)

        return df_fe

    def create_interaction_features(self, df):
        """Создание interaction features"""
        df_fe = df.copy()

        # Базовые взаимодействия
        interaction_pairs = [
            ("mileage", "torque"),
            ("km_driven", "mileage"),
            ("max_power", "max_torque_rpm"),
            ("engine", "max_power"),
            ("year", "mileage"),
            ("year", "max_power"),
            ("year", "engine"),
        ]

        for col1, col2 in interaction_pairs:
            if col1 in df_fe.columns and col2 in df_fe.columns:
                df_fe[f"{col1}_{col2}"] = df_fe[col1] * df_fe[col2]

        if "torque" in df_fe.columns and "max_torque_rpm" in df_fe.columns:
            df_fe["torque_rpm_ratio"] = df_fe["torque"] / (
                df_fe["max_torque_rpm"] + 1e-8
            )

        # Квадраты взаимодействий
        for feature in self.interaction_features:
            if feature in df_fe.columns:
                df_fe[f"{feature}_squared"] = df_fe[feature] ** 2

        return df_fe

    def _apply_all_feature_engineering(self, df):
        """Применение всего feature engineering"""
        df_fe = df.copy()
        df_fe = self.extract_brand_and_country(df_fe)
        df_fe = self.create_polynomial_features(df_fe)
        df_fe = self.create_log_features(df_fe)
        df_fe = self.create_interaction_features(df_fe)
        return df_fe

    def fit(self, df_train, target_col=None):
        """Обучение препроцессора на тренировочных данных"""
        if target_col and target_col in df_train.columns:
            df_train = df_train.drop([target_col], axis=1)

        # 1. Исправление типов
        df_processed = self.fix_column_types(df_train)

        # 2. Заполнение пропусков
        cols_with_nans = df_processed.columns[df_processed.isnull().any()].tolist()
        self.medians = {col: df_processed[col].median() for col in cols_with_nans}

        for col in cols_with_nans:
            df_processed[col] = df_processed[col].fillna(self.medians[col])

        if "engine" in df_processed.columns:
            df_processed["engine"] = df_processed["engine"].astype(int)
        if "seats" in df_processed.columns:
            df_processed["seats"] = df_processed["seats"].astype(int)

        # 3. Сохраняем статистики для численных features (до feature engineering)
        for feature in self.numerical_features:
            if feature in df_processed.columns:
                self.train_means[feature] = df_processed[feature].mean()
                self.train_stds[feature] = df_processed[feature].std()
                self.train_mins[feature] = df_processed[feature].min()

        # 4. Применяем ВЕСЬ feature engineering для обучения
        df_processed = self._apply_all_feature_engineering(df_processed)

        # 5. Обучаем OneHotEncoder
        categorical_cols = [
            "seats",
            "name",
            "fuel",
            "seller_type",
            "transmission",
            "owner",
            "country",
        ]
        available_categorical_cols = [
            col for col in categorical_cols if col in df_processed.columns
        ]

        self.ohe = OneHotEncoder(
            drop="first", dtype=int, sparse_output=False, handle_unknown="ignore"
        )
        self.ohe.fit(df_processed[available_categorical_cols])
        self.feature_names = self.ohe.get_feature_names_out(available_categorical_cols)

        # 6. Подготавливаем финальные фичи и обучаем scaler
        df_final = self._prepare_final_features(df_processed)

        # Сохраняем имена финальных фичей для проверки в transform
        self.final_feature_names_ = df_final.columns.tolist()

        self.scaler = StandardScaler()
        self.scaler.fit(df_final)

        return self

    def transform(self, df):
        """Применение преобразований к новым данным"""
        # 1. Исправление типов
        df_processed = self.fix_column_types(df)

        # 2. Заполнение пропусков
        for col, median in self.medians.items():
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(median)

        # 3. Приведение типов
        if "engine" in df_processed.columns:
            df_processed["engine"] = df_processed["engine"].astype(int)
        if "seats" in df_processed.columns:
            df_processed["seats"] = df_processed["seats"].astype(int)

        # 4. Применяем ВЕСЬ feature engineering (как в fit)
        df_processed = self._apply_all_feature_engineering(df_processed)

        # 5. Подготовка финальных features
        df_final = self._prepare_final_features(df_processed)

        # 6. Проверяем, что фичи совпадают с теми, на которых обучался scaler
        current_features = df_final.columns.tolist()
        if hasattr(self, "final_feature_names_"):
            missing_features = set(self.final_feature_names_) - set(current_features)
            extra_features = set(current_features) - set(self.final_feature_names_)

            if missing_features:
                print(
                    f"Предупреждение: отсутствуют фичи из обучения: {missing_features}"
                )
                # Добавляем недостающие фичи с нулевыми значениями
                for feature in missing_features:
                    df_final[feature] = 0

            if extra_features:
                print(f"Предупреждение: удаляем лишние фичи: {extra_features}")
                df_final = df_final[self.final_feature_names_]

        # 7. Убеждаемся, что порядок фичей такой же как при обучении
        if hasattr(self, "final_feature_names_"):
            df_final = df_final.reindex(columns=self.final_feature_names_, fill_value=0)

        # 8. Масштабирование
        if self.scaler is not None:
            df_scaled = pd.DataFrame(
                self.scaler.transform(df_final), columns=df_final.columns
            )
            return df_scaled
        else:
            return df_final

    def _prepare_final_features(self, df):
        """Подготовка финального набора features"""
        categorical_cols = [
            "seats",
            "name",
            "fuel",
            "seller_type",
            "transmission",
            "owner",
            "country",
        ]
        available_categorical_cols = [
            col for col in categorical_cols if col in df.columns
        ]

        # OneHotEncoding
        if self.ohe is not None and available_categorical_cols:
            cat_data = self.ohe.transform(df[available_categorical_cols])
            cat_df = pd.DataFrame(cat_data, columns=self.feature_names, index=df.index)
        else:
            cat_df = pd.DataFrame(index=df.index)

        # Численные признаки (все кроме категориальных)
        numerical_columns = [col for col in df.columns if col not in categorical_cols]
        num_df = df[numerical_columns]

        result_df = pd.concat([num_df, cat_df], axis=1)

        # Заполняем возможные NaN
        result_df = result_df.fillna(0)

        return result_df
