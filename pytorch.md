### 1. Подготовка
Если видеокарта NVidia поддерживает CUDA, установите [cuda toolkit](cuda_toolkit.md)

### 2. Установка pytorch
с поддержкой CUDA
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

без поддержки CUDA
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
### 3. Проверка
```python
import torch
x = torch.rand(5, 3)
print(x)
```

Вывод должен быть похож на:
```
tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])
```
Также проверьте, что GPU-драйвер и  CUDA (если есть) установлены и работают:
```python
import torch
torch.cuda.is_available()
```
