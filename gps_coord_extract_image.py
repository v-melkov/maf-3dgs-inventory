import sys
import os
import exifread
import re

def convert_exif_gps(gps_tag):
    """
    Преобразует тег GPS из формата exifread (кортеж дробей) в десятичные градусы.
    Пример: ((55, 1), (45, 1), (21, 1)) -> 55.755833...
    """
    try:
        # gps_tag обычно список из трёх пар (числитель, знаменатель)
        if len(gps_tag) != 3:
            return None
        
        degrees = float(gps_tag[0].num) / float(gps_tag[0].den)
        minutes = float(gps_tag[1].num) / float(gps_tag[1].den)
        seconds = float(gps_tag[2].num) / float(gps_tag[2].den)
        
        return degrees + (minutes / 60.0) + (seconds / 3600.0)
    except Exception:
        return None

def extract_gps_from_image(image_path):
    """
    Извлекает GPS-координаты из изображения, используя библиотеку exifread.
    """
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл '{image_path}' не найден.")
        return None

    try:
        # Открываем файл в бинарном режиме
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        
        # Ищем GPS-теги
        gps_lat = tags.get('GPS GPSLatitude')
        gps_lon = tags.get('GPS GPSLongitude')
        gps_lat_ref = tags.get('GPS GPSLatitudeRef')
        gps_lon_ref = tags.get('GPS GPSLongitudeRef')
        gps_alt = tags.get('GPS GPSAltitude')
        gps_alt_ref = tags.get('GPS GPSAltitudeRef')  # 0 = выше уровня моря, 1 = ниже

        if not (gps_lat and gps_lon and gps_lat_ref and gps_lon_ref):
            print("В метаданных изображения не найдены GPS-координаты.")
            print("\nДоступные теги (первые 20):")
            count = 0
            for key, value in tags.items():
                if count >= 20:
                    break
                print(f"  {key}: {value}")
                count += 1
            return None

        # Преобразуем координаты в десятичные градусы
        lat = convert_exif_gps(gps_lat.values)
        lon = convert_exif_gps(gps_lon.values)
        if lat is None or lon is None:
            print("Не удалось разобрать координаты (неверный формат).")
            return None

        # Учитываем направление
        lat_ref_str = str(gps_lat_ref).strip()
        lon_ref_str = str(gps_lon_ref).strip()
        if lat_ref_str.upper() == 'S':
            lat = -lat
        if lon_ref_str.upper() == 'W':
            lon = -lon

        # Высота (необязательно)
        altitude = None
        if gps_alt:
            try:
                alt_val = float(gps_alt.num) / float(gps_alt.den)
                # Если есть AltitudeRef, корректируем знак
                if gps_alt_ref:
                    alt_ref = int(str(gps_alt_ref).strip())
                    if alt_ref == 1:  # ниже уровня моря
                        alt_val = -alt_val
                altitude = alt_val
            except Exception:
                pass

        return {
            'latitude': lat,
            'longitude': lon,
            'altitude': altitude,
            'raw': f"{gps_lat.values} {gps_lat_ref}, {gps_lon.values} {gps_lon_ref}"
        }

    except Exception as e:
        print(f"Произошла ошибка при обработке файла: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python extract_gps_image.py <путь_к_изображению>")
        sys.exit(1)

    image_file = sys.argv[1]
    result = extract_gps_from_image(image_file)

    if result:
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ ИЗВЛЕЧЕНИЯ GPS (из изображения)")
        print("="*50)
        print(f"Широта (Latitude):  {result['latitude']:.6f}°")
        print(f"Долгота (Longitude): {result['longitude']:.6f}°")
        if result.get('altitude') is not None:
            print(f"Высота (Altitude):   {result['altitude']:.2f} м")
        print("\nИсходные данные:")
        print(f"  {result.get('raw', '')}")
        print("="*50)

        print(f"\n📌 Для вставки в QGIS (координаты в десятичных градусах):")
        print(f"   {result['latitude']}, {result['longitude']}")
    else:
        print("Не удалось извлечь GPS-координаты из изображения.")