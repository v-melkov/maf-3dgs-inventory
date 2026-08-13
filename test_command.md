# Один раз в начале сессии, из папки проекта
export MAF_ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$MAF_ROOT"
export VIDEO="00_source/IMG_2048.MOV"
mkdir -p reports

# Проверка видео
ffprobe -hide_banner -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,duration,nb_frames,codec_name \
  -of default=noprint_wrappers=1 "$VIDEO"
ffprobe -hide_banner -show_entries format=duration,bit_rate -of default=noprint_wrappers=1 "$VIDEO"
exiftool -ee -G1 -a "$VIDEO" | grep -iE 'gps|projection|location'

# Извлечение кадров
sharp-frames "$VIDEO" 02_frames --fps 30 --num-frames 150 --format jpg --force-overwrite

# Геотегирование
python -m maf_pipeline.st04_geotag_stub 02_frames --mode anchor --video "$VIDEO" --report reports/st04.json
exiftool -T -FileName -GPSLatitude -GPSLongitude -DateTimeOriginal -n 02_frames/*.jpg > reports/st04_exif.tsv
exiftool -if 'not $GPSLatitude' -p '$FileName' 02_frames/*.jpg      # кадры без координат

# COLMAP
python -m maf_pipeline.st05_sfm_pycolmap 02_frames 03_colmap --max-image-size 1600 --matcher sequential --report reports/st05.json

# Масштабирование
python -m maf_pipeline.st08_scale 02_frames 03_colmap/sparse/0_txt --project project.json --report reports/st08.json
