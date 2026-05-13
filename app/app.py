import os
import streamlit as st
import tempfile
import pandas as pd
import python_benchmark

# Список способов переноса/оптимизации модели (как в оригинале)
methods = (
    "Исходная модель (Python)",
    "ONNX Runtime",
    "TensorRT",
    "OpenVINO",
    "TFLite",
    "PyTorch (TorchScript)",
    "TensorFlow (SavedModel)"
)

st.set_page_config(page_title="Оптимизатор инференса ML-модели", layout="centered")
st.title("Оптимизатор инференса ML-модели")

# Загрузка файлов
model_file = st.file_uploader("ML-модель", type=["pkl"], accept_multiple_files=False)
dataset_file = st.file_uploader("Датасет", type=["csv"], accept_multiple_files=False)

# Кнопка запуска сравнения
if st.button("Загрузить и выполнить сравнение"):
    if model_file is None or dataset_file is None:
        st.error("Пожалуйста, загрузите и модель (.pkl), и датасет (.csv).")
    else:
        # Сохраняем загруженные файлы во временные файлы
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_model:
            tmp_model.write(model_file.getbuffer())
            tmp_model_path = tmp_model.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_dataset:
            tmp_dataset.write(dataset_file.getbuffer())
            tmp_dataset_path = tmp_dataset.name

        results = []
        try:
            for method in methods:
                if method == "Исходная модель (Python)":
                    # Вызов внешних скриптов
                    try:
                        # Получение времени (мс)
                        time_val = python_benchmark.get_time(tmp_model_path, tmp_dataset_path)
                    except Exception:
                        #time_val = None
                    try:
                        # Получение памяти (МБ)
                        mem_val = python_benchmark.get_memory(tmp_model_path, tmp_dataset_path)
                    except Exception:
                        mem_val = None
                    try:
                        # Получение точности (%)
                        acc_val = python_benchmark.get_accuracy(tmp_model_path, tmp_dataset_path)
                    except Exception:
                        acc_val = None
                else:
                    time_val = 0.0
                    mem_val = 0.0
                    acc_val = 0.0

                results.append({
                    "Способ переноса / Оптимизация": method,
                    "Скорость (мс)": time_val if time_val is not None else "Ошибка",
                    "Память (МБ)": mem_val if mem_val is not None else "Ошибка",
                    "Точность (%)": acc_val if acc_val is not None else "Ошибка"
                })
        finally:
            # Удаляем временные файлы
            os.unlink(tmp_model_path)
            os.unlink(tmp_dataset_path)

        # Отображаем таблицу результатов
        st.subheader("Сравнение методов оптимизации инференса")
        df = pd.DataFrame(results)

        for col in ("Скорость (мс)", "Память (МБ)", "Точность (%)"):
            if col == "Скорость (мс)":
                df[col] = df[col].apply(
                lambda x: x if x == "Ошибка" else f"{round(x)}"
                )
            else:
                df[col] = df[col].apply(
                lambda x: x if x == "Ошибка" else f"{x:.2f}"
                )

        st.dataframe(df, width="stretch", hide_index=True)
else:
    st.info("Загрузите модель и датасет, затем нажмите кнопку.")
