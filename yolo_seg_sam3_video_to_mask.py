from ultralytics.models.sam import SAM3VideoSemanticPredictor

# Инициализация видео-предиктора
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    imgsz=644,
    model="sam3.pt",
#    quantize=16,
    save=False,
)
predictor = SAM3VideoSemanticPredictor(overrides=overrides)

# Сегментация и отслеживание объектов в видео
results = predictor(
    source="samples/IMG_1544.MOV",
    text=["chair"],
    stream=True  # Потоковая обработка кадров
)

# Обработка результатов по кадрам
for r in results:
    r.show()  # Показать кадр с масками
    #r.save()  # Раскомментируйте для сохранения каждого кадра