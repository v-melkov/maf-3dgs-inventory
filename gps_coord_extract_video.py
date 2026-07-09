import ffmpeg
import json
import sys
import os
import re

def extract_gps_with_ffmpeg(video_path):
    """
    Извлекает GPS-координаты из видео, используя ffmpeg-python и ffprobe.
    """
    if not os.path.exists(video_path):
        print(f"Ошибка: Файл '{video_path}' не найден.")
        return None

    try:
        # Извлекаем все метаданные с помощью ffprobe
        # format='json' гарантирует получение структурированных данных
        metadata = ffmpeg.probe(video_path)
        
        # Метаданные могут находиться в разных местах.
        # 1. Сначала ищем в глобальных метаданных формата (format)
        gps_data = {}
        format_tags = metadata.get('format', {}).get('tags', {})
        
        # Пробуем найти координаты в формате "композитной" строки от Apple
        if 'com.apple.quicktime.location.ISO6709' in format_tags:
            location_str = format_tags['com.apple.quicktime.location.ISO6709']
            # Строка имеет вид: "+55.7558+037.6173+156.000/"
            gps_data = parse_iso6709(location_str)
            if gps_data:
                return gps_data
        
        # 2. Если не нашли, ищем в стандартных тегах GPS
        # Они могут быть как в формате, так и в потоках (streams)
        for stream in metadata.get('streams', []):
            stream_tags = stream.get('tags', {})
            # Объединяем теги из stream и format для поиска
            all_tags = {**format_tags, **stream_tags}
            
            lat = all_tags.get('GPSLatitude')
            lon = all_tags.get('GPSLongitude')
            lat_ref = all_tags.get('GPSLatitudeRef')
            lon_ref = all_tags.get('GPSLongitudeRef')
            
            if lat and lon:
                # Пробуем преобразовать, если это строка типа "55/1 45/1 21/1"
                gps_data = convert_ffmpeg_gps(lat, lon, lat_ref, lon_ref)
                if gps_data:
                    return gps_data
        
        # 3. Если координаты не найдены, выводим все доступные теги для отладки
        print("В метаданных видео не найдены GPS-координаты.")
        print("\nДоступные метаданные (первые 20 полей):")
        all_tags = format_tags
        for stream in metadata.get('streams', []):
            all_tags.update(stream.get('tags', {}))
        
        count = 0
        for key, value in all_tags.items():
            if count >= 20:
                break
            print(f"  {key}: {value}")
            count += 1
        
        return None
        
    except ffmpeg.Error as e:
        print(f"Ошибка ffmpeg: {e.stderr.decode()}")
        return None
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return None

def parse_iso6709(location_str):
    """
    Парсит строку формата ISO 6709, которую Apple использует для хранения координат.
    Пример: "+55.7558+037.6173+156.000/"
    """
    # Убираем слеш в конце, если есть
    location_str = location_str.rstrip('/')
    
    # Регулярное выражение для поиска широты, долготы и высоты
    # Широта: [+-]дробное число
    # Долгота: [+-]дробное число
    # Высота (опционально): [+-]дробное число
    pattern = r'([+-]\d+\.\d+)([+-]\d+\.\d+)([+-]\d+\.\d+)?'
    match = re.match(pattern, location_str)
    
    if match:
        lat = float(match.group(1))
        lon = float(match.group(2))
        alt = float(match.group(3)) if match.group(3) else None
        
        return {
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'raw': location_str
        }
    return None

def convert_ffmpeg_gps(lat_str, lon_str, lat_ref, lon_ref):
    """
    Преобразует GPS-координаты из формата ffprobe в десятичные градусы.
    ffprobe часто возвращает их в виде "55/1 45/1 21/1" (градусы/1 минуты/1 секунды/1)
    """
    try:
        lat_parts = parse_ffmpeg_coordinate(lat_str)
        lon_parts = parse_ffmpeg_coordinate(lon_str)
        
        if not lat_parts or not lon_parts:
            return None
        
        lat_deg, lat_min, lat_sec = lat_parts
        lon_deg, lon_min, lon_sec = lon_parts
        
        lat = lat_deg + (lat_min / 60.0) + (lat_sec / 3600.0)
        lon = lon_deg + (lon_min / 60.0) + (lon_sec / 3600.0)
        
        if lat_ref and lat_ref.upper() == 'S':
            lat = -lat
        if lon_ref and lon_ref.upper() == 'W':
            lon = -lon
        
        return {
            'latitude': lat,
            'longitude': lon,
            'altitude': None,  # ffprobe обычно не возвращает высоту в этом формате
            'raw': f"{lat_str}, {lon_str}"
        }
    except Exception:
        return None

def parse_ffmpeg_coordinate(coord_str):
    """
    Парсит строку координаты от ffprobe вида "55/1 45/1 21/1"
    Возвращает кортеж (градусы, минуты, секунды)
    """
    parts = coord_str.strip().split()
    if len(parts) != 3:
        return None
    
    try:
        deg = float(parts[0].split('/')[0]) / float(parts[0].split('/')[1])
        minutes = float(parts[1].split('/')[0]) / float(parts[1].split('/')[1])
        seconds = float(parts[2].split('/')[0]) / float(parts[2].split('/')[1])
        return (deg, minutes, seconds)
    except (ValueError, ZeroDivisionError, IndexError):
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python extract_gps_ffmpeg.py <путь_к_видео>")
        sys.exit(1)
    
    video_file = sys.argv[1]
    result = extract_gps_with_ffmpeg(video_file)
    
    if result:
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ ИЗВЛЕЧЕНИЯ GPS (через ffmpeg)")
        print("="*50)
        print(f"Широта (Latitude):  {result['latitude']:.6f}°")
        print(f"Долгота (Longitude): {result['longitude']:.6f}°")
        if result.get('altitude'):
            print(f"Высота (Altitude):   {result['altitude']} м")
        print("\nИсходные данные:")
        print(f"  {result.get('raw', '')}")
        print("="*50)
        
        print(f"\n📌 Для вставки в QGIS (координаты в десятичных градусах):")
        print(f"   {result['latitude']}, {result['longitude']}")
    else:
        print("Не удалось извлечь GPS-координаты из видео.")