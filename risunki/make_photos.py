"""Сборка фото-иллюстраций для ВКР из отобранных кадров и рендеров."""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

UP = '/mnt/user-data/uploads'
REPO = f'{UP}/maf-3dgs-inventory'
SEL = '/tmp/claude-0/sel'
OUT = '/mnt/user-data/outputs/risunki/foto'
os.makedirs(OUT, exist_ok=True)

FONT = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
FONTB = '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'


def font(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, sz)


def load(path):
    im = Image.open(path)
    if im.mode not in ('RGB',):
        if im.mode.startswith('I;16') or im.mode == 'I':
            arr = np.array(im)
            im = Image.fromarray((arr >> 8).astype('uint8'))
        im = im.convert('RGB')
    return im


def crop_frac(im, l, t, r, b):
    w, h = im.size
    return im.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


def fit_h(im, h):
    return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)


def panel_row(images, labels, height=700, gap=24, label_h=None, fs=None):
    """Горизонтальный ряд панелей с подписями «а) …» под каждой."""
    fs = fs or max(26, round(height * 0.05))
    nl = max(l.count('\n') + 1 for l in labels)
    label_h = label_h or 20 + nl * (fs + 8)
    ims = [fit_h(i, height) for i in images]
    W = sum(i.width for i in ims) + gap * (len(ims) - 1)
    H = height + label_h
    canvas = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(canvas)
    x = 0
    for im, lab in zip(ims, labels):
        canvas.paste(im, (x, 0))
        d.rectangle([x, 0, x + im.width - 1, height - 1], outline=(150, 150, 150), width=2)
        lines = lab.split('\n')
        for k, ln in enumerate(lines):
            tw = d.textlength(ln, font=font(fs))
            d.text((x + (im.width - tw) / 2, height + 12 + k * (fs + 6)), ln, fill=(26, 26, 26), font=font(fs))
        x += im.width + gap
    return canvas


def grid2x2(images, labels, cell=(900, 640), gap=24, label_h=56, fs=30):
    cw, ch = cell
    W = cw * 2 + gap
    H = (ch + label_h) * 2 + gap
    canvas = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(canvas)
    for i, (im, lab) in enumerate(zip(images, labels)):
        r, c = divmod(i, 2)
        # cover-fit into the cell
        s = max(cw / im.width, ch / im.height)
        im2 = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        ox, oy = (im2.width - cw) // 2, (im2.height - ch) // 2
        im2 = im2.crop((ox, oy, ox + cw, oy + ch))
        x, y = c * (cw + gap), r * (ch + label_h + gap)
        canvas.paste(im2, (x, y))
        d.rectangle([x, y, x + cw - 1, y + ch - 1], outline=(150, 150, 150), width=2)
        tw = d.textlength(lab, font=font(fs))
        d.text((x + (cw - tw) / 2, y + ch + 12), lab, fill=(26, 26, 26), font=font(fs))
    return canvas


def save(im, name):
    p = f'{OUT}/{name}.jpg'
    im.save(p, quality=92, dpi=(300, 300))
    print('saved', p, im.size)


# ---------------------------------------------------------------- рендеры SuperSplat
m = {i: load(f'{SEL}/pdfimg/m-00{i}.png') for i in range(6)}
# убрать элементы интерфейса просмотрщика (рамка, навигация, панель кнопок)
bench_scene = crop_frac(m[0], 0.035, 0.11, 0.965, 0.90)
bench_seg = crop_frac(m[1], 0.0, 0.03, 1.0, 0.97)
dino1_r = crop_frac(m[2], 0.03, 0.02, 0.97, 0.90)
dino2_r = crop_frac(m[3], 0.0, 0.0, 1.0, 0.94)
planet_r = crop_frac(m[4], 0.0, 0.0, 1.0, 0.94)
sun_r = crop_frac(m[5], 0.0, 0.0, 1.0, 0.94)

# ---------------------------------------------------------------- 1. классы сложности
bench_ph = load(f'{UP}/skameyka1/images/frame_01640.jpeg')
cont_ph = load(f'{UP}/3dgs/musorka/musorka2_00062.jpeg')
dino_ph = load(f'{UP}/dino2/frame_00010.png')
bench_c = crop_frac(bench_ph, 0.08, 0.10, 0.92, 0.98)
cont_c = crop_frac(cont_ph, 0.0, 0.14, 1.0, 0.80)
dino_c = crop_frac(dino_ph, 0.0, 0.0, 1.0, 1.0)
save(panel_row([bench_c, cont_c, dino_c],
               ['а) низкая сложность:\nскамейка с урной',
                'б) средняя сложность:\nконтейнерная площадка',
                'в) высокая сложность:\nпарковая скульптура']),
     'foto_2_1_2_klassy_slozhnosti')

# ---------------------------------------------------------------- 2. сложные поверхности
slats = crop_frac(bench_ph, 0.12, 0.24, 0.62, 0.70)
bins = crop_frac(load(f'{UP}/musorka/images/frame_00030.jpg'), 0.05, 0.38, 1.0, 0.72)
chains = crop_frac(load(f'{UP}/dino1/frame_01086.jpeg'), 0.20, 0.25, 0.85, 0.95)
sun = crop_frac(sun_r, 0.18, 0.08, 0.82, 0.92)
save(grid2x2([slats, bins, chains, sun],
             ['а) тонкие рейки и зазоры между ними',
              'б) глянцевый пластик, колёса, тени',
              'в) цепи, сетка, тонкие стойки',
              'г) блики на гладкой сфере (3DGS-модель)']),
     'foto_2_1_3_slozhnye_poverhnosti')

# ---------------------------------------------------------------- 3. эталон с маркерами
fr = cv2.imread(f'{UP}/3dgs/aruco_test/musorka2_00249.jpeg')
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
                              cv2.aruco.DetectorParameters())
corners, ids, _ = det.detectMarkers(fr)
side = np.mean([cv2.arcLength(c.reshape(-1, 1, 2), True) / 4 for c in corners])
lw = max(3, int(side * 0.03))
vis = fr.copy()
for c in corners:
    pts = c.reshape(-1, 2).astype(int)
    cv2.polylines(vis, [pts], True, (40, 40, 200), lw, cv2.LINE_AA)
    for p in pts:
        cv2.circle(vis, tuple(p), int(lw * 1.6), (0, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(vis, tuple(pts.mean(0).astype(int)), int(lw * 1.4), (40, 40, 200), -1, cv2.LINE_AA)
vis_rgb = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
vis_clean = vis_rgb.copy()
d = ImageDraw.Draw(vis_rgb)
fsz = int(side * 0.45)
for c, i in zip(corners, ids.ravel()):
    pts = c.reshape(-1, 2)
    lab = f'ID {i}'
    f = font(fsz, True)
    tw = d.textlength(lab, font=f)
    x = pts[:, 0].min() - tw - fsz * 0.9          # подпись слева от маркера
    y = pts[:, 1].mean() - fsz * 0.6
    d.rectangle([x - 12, y - 8, x + tw + 12, y + fsz * 1.2], fill=(255, 255, 255), outline=(200, 40, 40), width=max(3, lw // 2))
    d.text((x, y), lab, fill=(200, 40, 40), font=f)
et = vis_rgb
ib = list(ids.ravel()).index(min(ids.ravel()))
zc = corners[ib].reshape(-1, 2)
zx, zy = zc[:, 0].mean(), zc[:, 1].mean(); r = side * 1.05
zoom = vis_clean.crop((int(zx - r), int(zy - r), int(zx + r), int(zy + r)))
save(panel_row([et, zoom],
               ['а) рейка с маркерами\nна концах',
                f'б) фрагмент: углы и центр\nмаркера ID {ids.ravel()[ib]}'], height=1000),
     'foto_2_5_3_etalon_aruco')
print('маркеры:', ids.ravel().tolist(), 'сторона, px:', round(side))

# ---------------------------------------------------------------- 4. сегментация
save(panel_row([bench_scene, bench_seg],
               ['а) модель, обученная по полным кадрам',
                'б) модель, обученная с масками объекта'], height=620),
     'foto_2_4_6_segmentaciya_3dgs')

# ---------------------------------------------------------------- 5. визуальное качество
save(panel_row([dino1_r, dino2_r],
               ['а) скульптурная композиция «гнездо»',
                'б) скульптура динозавра'], height=620),
     'foto_3_2_5_kachestvo_modeley')

# ---------------------------------------------------------------- 6. блики (планетарий)
save(panel_row([planet_r, sun_r],
               ['а) общий вид сцены',
                'б) отражения на глянцевой сфере'], height=620),
     'foto_2_1_1_bliki_planetariy')

# ---------------------------------------------------------------- 7. траектории камеры (COLMAP)
TV = f'{REPO}/risunki/_prosmotr'
def trim(im):
    from PIL import ImageChops
    bg = Image.new(im.mode, im.size, (255, 255, 255))
    bb = ImageChops.difference(im, bg).getbbox()
    return im.crop(bb) if bb else im
tva = trim(load(f'{TV}/tv_a.png'))
tvb = trim(load(f'{TV}/tv_b.png'))
save(panel_row([tva, tvb],
               ['а) скамейка с урной: три обхода\nна разном удалении от объекта',
                'б) контейнерная площадка: один обход,\nсектор между началом и концом не покрыт'],
               height=1100),
     'foto_2_4_5_traektorii_kamery')

# ---------------------------------------------------------------- 8. скриншоты приложений
SC = f'{REPO}/risunki/scriny'
def sc(n):
    return load(f'{SC}/Screenshot_{n}.png')
LF_VIEW = (380, 40, 1624, 1014)        # область просмотра LichtFeld Studio
CM_VIEW = (180, 150, 1290, 950)        # область просмотра COLMAP (рабочая часть сцены)
LF_WIN = (0, 0, 1920, 1022)            # окно LichtFeld без панели задач Windows

# 8.1 COLMAP: последовательная регистрация изображений
save(panel_row([sc(6).crop(CM_VIEW), sc(8).crop(CM_VIEW), sc(10).crop(CM_VIEW)],
               ['а) 14 изображений,\n5 685 точек',
                'б) 81 изображение,\n30 344 точки',
                'в) 150 изображений,\n47 906 точек']),
     'skrin_2_4_5_colmap_registraciya')

# 8.2 LichtFeld: ход обучения
save(grid2x2([sc(12).crop(LF_VIEW), sc(13).crop(LF_VIEW), sc(14).crop(LF_VIEW), sc(18).crop(LF_VIEW)],
             ['а) итерация 60: 47 906 гауссианов',
              'б) итерация 320: 125 379 гауссианов',
              'в) итерация 950: 304 749 гауссианов',
              'г) итерация 3 170: 4 625 917 гауссианов'], cell=(900, 700)),
     'skrin_2_4_7_obuchenie_3dgs')

# 8.3 LichtFeld: интерфейс среды с загруженным набором данных
save(panel_row([sc(16).crop(LF_WIN)], [''], height=1022, label_h=1),
     'skrin_2_3_lichtfeld_interfeys')

# 8.4 качество в центре и на периферии сцены
save(panel_row([sc(3).crop(LF_VIEW), sc(4).crop(LF_VIEW)],
               ['а) объект в зоне полного покрытия',
                'б) периферия сцены: размытые стойки\nи «плавающие» гауссианы']),
     'skrin_3_2_5_artefakty_periferii')

# 8.5 LichtFeld: импорт набора данных COLMAP
save(panel_row([sc(15).crop((660, 255, 1240, 800))], [''], height=1000, label_h=1),
     'skrin_3_1_7_import_dannyh')
