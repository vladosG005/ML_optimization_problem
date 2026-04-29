from sys import argv
from load_model import load_model
import resource

import psutil
import os
import time
import threading
X, Y, model = load_model(argv[1], argv[2])
class PeakMemoryMonitor:
    def __init__(self):
        self.max_mem = 0
        self.running = True
        self.process = psutil.Process(os.getpid())

    def _monitor(self):
        while self.running:
            try:
                # Считаем RSS текущего процесса + всех детей (если они есть)
                current_mem = self.process.memory_info().rss
                for child in self.process.children(recursive=True):
                    current_mem += child.memory_info().rss
                
                if current_mem > self.max_mem:
                    self.max_mem = current_mem
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.01)  # Проверка каждые 10 мс

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.thread.join()
        print(f"Пиковое потребление RAM: {self.max_mem / 1024**2:.2f} MB")

# Использование:
with PeakMemoryMonitor():
    results = model.predict(X)
