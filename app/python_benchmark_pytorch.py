import sys
import time
import gc
import numpy as np
import torch
from hummingbird.ml import convert
from sklearn.metrics import accuracy_score

# Кэш для скомпилированных TorchScript моделей
_torch_cache = {}

def _get_torch_model(model, X):
    """Конвертирует sklearn/joblib модель в TorchScript с кэшированием."""
    model_id = id(model)
    if model_id not in _torch_cache:
        X_sample = np.ascontiguousarray(X[:min(len(X), 10)], dtype=np.float32)
        
        hb_model = convert(
            model, 
            "torch", 
            test_input=X_sample, 
            extra_config={"tree_implementation": "tree_trav"}
        )
        torch_module = hb_model.model
        torch_module.eval()
        
        dummy = torch.from_numpy(X_sample)
        traced = torch.jit.trace(torch_module, dummy)
        _torch_cache[model_id] = traced
        
    return _torch_cache[model_id]

def get_time(X, model):
    """Измеряет время инференса всего датасета в мс."""
    X_t = np.ascontiguousarray(X, dtype=np.float32)
    X_tensor = torch.from_numpy(X_t)
    torch_model = _get_torch_model(model, X)

    with torch.no_grad():
        _ = torch_model(X_tensor)  # Warmup

    start = time.perf_counter_ns()
    with torch.no_grad():
        _ = torch_model(X_tensor)
    end = time.perf_counter_ns()
    
    return round((end - start) / 1_000_000, 2)

def get_memory(X, model):
    """Измеряет пиковую RSS-память процесса после инференса."""
    X_t = np.ascontiguousarray(X, dtype=np.float32)
    X_tensor = torch.from_numpy(X_t)
    torch_model = _get_torch_model(model, X)

    with torch.no_grad():
        _ = torch_model(X_tensor)

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
    X_t = np.ascontiguousarray(X, dtype=np.float32)
    X_tensor = torch.from_numpy(X_t)
    torch_model = _get_torch_model(model, X)

    with torch.no_grad():
        output = torch_model(X_tensor)

    #1. Распаковка кортежей (Hummingbird часто возвращает (preds, probs))
    if isinstance(output, (tuple, list)):
        output = output[0]

    #2. Приведение к numpy
    if isinstance(output, torch.Tensor):
        preds = output.cpu().numpy()
    else:
        preds = np.asarray(output)

    #3. Строгое выравнивание до 1D
    preds = np.asarray(preds).ravel()
    Y_flat = np.asarray(Y).ravel()

    #4. Конвертация вероятностей -> метки классов
    if preds.dtype.kind in ('f',):  # float32/64
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = (preds >= 0.5).astype(int)
        elif preds.ndim == 2 and preds.shape[1] > 1:
            preds = np.argmax(preds, axis=1)
        elif preds.ndim == 1:
            preds = (preds >= 0.5).astype(int)

    #5. Синхронизация типов: если Y строки, а preds числа → мапим обратно
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
    """Очищает кэш моделей между запусками."""
    global _torch_cache
    _torch_cache = {}
    gc.collect()
    torch.cuda.empty_cache()