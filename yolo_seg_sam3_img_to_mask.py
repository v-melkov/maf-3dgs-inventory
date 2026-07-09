from ultralytics.models.sam import SAM3SemanticPredictor
import cv2
import numpy as np

# Инициализация предиктора
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",  # Убедитесь, что файл весов скачан
#    quantize=16,      # FP16 для ускорения !!!включить для GPU!!!
    save=True,        # Сохранять визуализацию (картинку с наложенной маской)
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Загрузка изображения
predictor.set_image("samples/IMG_1544/frame-00000.png")

# Сегментация урн и скамеек
results = predictor(text=["trash can", "bench", "chair"])

# Обработка результатов
for i, result in enumerate(results):
    # 1. Сохраняем красивую картинку с наложением (как было раньше)
    result.save(filename=f"samples/visualization_{i}.jpg")
    
    # 2. Сохраняем ЧИСТУЮ маску (черно-белую)
    if result.masks is not None:
        # Получаем данные масок (тензор)
        masks_data = result.masks.data
        
        # Проходим по каждой найденной маске в этом результате
        for j, mask_tensor in enumerate(masks_data):
            # Переводим в NumPy массив и меняем диапазон с [0, 1] на [0, 255]
            mask_np = mask_tensor.cpu().numpy()
            mask_uint8 = (mask_np * 255).astype(np.uint8)
            
            # Сохраняем как PNG (лучше для черно-белых изображений без потерь)
            filename = f"mask_object_{i}_part_{j}.png"
            cv2.imwrite(filename, mask_uint8)
            print(f"Чистая маска сохранена: {filename}")
    else:
        print(f"Маски не найдены для результата {i}")