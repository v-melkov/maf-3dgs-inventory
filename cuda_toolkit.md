### 1. Подготовка

*   **Windows:** Версия 11 или 10 (21H2 и выше).
*   **Видеокарта:** Совместимая с CUDA NVIDIA.
*   **WSL:** Ubuntu (рекомендуется 22.04 или 24.04).

### **2. Установка драйвера в Windows**

Установите только один драйвер в *Windows*. Он автоматически будет использоваться WSL. Не устанавливайте никаких драйверов внутри Ubuntu.


### **3. Установка CUDA Toolkit в Ubuntu**

Откройте терминал вашего дистрибутива Ubuntu в WSL и выполните команды.

```bash
wget https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda_13.2.0_595.45.04_linux.run
sudo sh cuda_13.2.0_595.45.04_linux.run
```


### **4. Настройка переменных окружения**

Чтобы команды `nvcc` и другие утилиты CUDA были доступны, добавьте пути в `~/.bashrc`:

```bash
echo 'export CUDA_PATH=/usr/local/cuda-13.2/' >> ~/.bashrc
echo 'export PATH=$CUDA_PATH/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
*(Примечание: если вы установили другую версию, например 12.8, путь может быть `/usr/local/cuda-12.8`)*.

### **5. Проверка установки**

Закройте и снова откройте терминал Ubuntu, затем выполните:

```bash
nvcc --version
```
Если вы видите информацию о версии CUDA, установка прошла успешно.