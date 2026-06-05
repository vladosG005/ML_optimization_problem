import sys
import time
import gc
import numpy as np
from sklearn.metrics import accuracy_score
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Кэш для скомпилированных ONNX-сессий
_onnx_sessions = {}

def _get_onnx_session(model, X):
    """Конвертирует sklearn модель в ONNX и создаёт сессию с кэшированием."""
    model_id = id(model)
    if model_id not in _onnx_sessions:
        initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        session = ort.InferenceSession(onnx_model.SerializeToString())
        _onnx_sessions[model_id] = session
    return _onnx_sessions[model_id]

def get_time(X, model):
    """Измеряет время инференса всего датасета в мс."""
    X_ort = X.astype(np.float32)
    session = _get_onnx_session(model, X)

    input_name = session.get_inputs()[0].name
    _ = session.run(None, {input_name: X_ort})  # Warmup

    start = time.perf_counter_ns()
    _ = session.run(None, {input_name: X_ort})
    end = time.perf_counter_ns()

    return round((end - start) / 1_000_000, 2)

def get_memory(X, model):
    """Измеряет пиковую RSS-память процесса после инференса."""
    X_ort = X.astype(np.float32)
    session = _get_onnx_session(model, X)

    input_name = session.get_inputs()[0].name
    _ = session.run(None, {input_name: X_ort})

    gc.collect()

    if sys.platform == "win32":
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return None
    else:
        import resource
        mem_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        mem_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 if sys.platform.startswith("linux") else 1024 * 1024
        return (mem_children + mem_self) / divisor

def get_accuracy(X, Y, model):
    """Вычисляет точность классификации в % с защитой от shape/type/corner-case ошибок."""
    X_ort = X.astype(np.float32)
    session = _get_onnx_session(model, X)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: X_ort})
    # ONNX может возвращать список тензоров: [predictions, probabilities]
    preds_raw = output[0]

    #1. Приведение к numpy
    preds = np.asarray(preds_raw)

    #2. Строгое выравнивание до 1D (только если 1D)
    if preds.ndim > 1 and preds.shape[1] == 1:
        preds = preds.ravel()
    elif preds.ndim > 1 and preds.shape[1] > 1:
        # Многокласс: берем аргмакс
        preds = np.argmax(preds, axis=1)

    #3. Синхронизация типов: если Y строки, а preds числа → мапим обратно
    Y_flat = np.asarray(Y).ravel()
    if Y_flat.dtype.kind in ('U', 'O') and preds.dtype.kind in ('i', 'f'):
        if hasattr(model, 'classes_'):
            try:
                preds = model.classes_[preds.astype(int)]
            except IndexError:
                pass
        else:
            try:
                Y_flat = Y_flat.astype(preds.dtype)
            except ValueError:
                return 0.0

    try:
        return round(100 * accuracy_score(Y_flat, preds), 2)
    except Exception:
        return 0.0

def clear_cache():
    """Очищает кэш сессий между запусками."""
    global _onnx_sessions
    _onnx_sessions = {}
    gc.collect()
