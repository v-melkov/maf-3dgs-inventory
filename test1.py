import os
import glob
from ultralytics.models.sam import SAM3SemanticPredictor

# Инициализация предиктора
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",
    quantize=16,
    save=True,  # Автоматически сохранять результаты
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Получаем список всех изображений в папке
image_folder = "samples/img_extracted/IMG_1544"
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))  # или *.png

# Обрабатываем каждое изображение
for image_path in sorted(image_files):
    print(f"Обработка: {image_path}")
    
    # Загружаем изображение
    predictor.set_image(image_path)
    
    # Сегментируем урны и скамейки
    results = predictor(text=["кресло"])
    
    # Результаты автоматически сохранятся благодаря save=True
    # Или можно сохранить вручную:
    # for result in results:
    #     result.save(filename=f"output/{os.path.basename(image_path)}")