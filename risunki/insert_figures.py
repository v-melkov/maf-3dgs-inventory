"""Расстановка рисунков в главах ВКР.

Для каждого рисунка задан якорь — начало абзаца (или строки), после которого
вставляется рисунок. В абзац-якорь добавляется ссылка «(рисунок N)»; если
якорь не абзац (таблица, заголовок), вставляется отдельная фраза-ссылка.
Нумерация сквозная по главам 1→2→3 в порядке следования в тексте.

Запуск: python3 insert_figures.py <каталог с glava1.md…glava3.md>
Повторный запуск безопасен: прежние вставки (блоки <!-- fig:N -->, ссылки
«(рисунок N)» и фразы-ссылки) сначала снимаются с текущего текста, затем
рисунки расставляются заново. Правки автора в остальном тексте сохраняются.
"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
MARK = '<!-- fig:'

# (файл, якорь-префикс, режим, путь к картинке, подпись, фраза-ссылка)
#  режим 'para'  — дописать «(рисунок N)» в абзац-якорь;
#  режим 'after' — якорь не абзац; вставить фразу-ссылку отдельным абзацем;
#  режим 'before' — то же, но перед строкой-якорем (обычно заголовком).
FIGS = [
    ('glava1.md', 'Три перечисленных уровня неравноценны', 'para',
     'risunki/fig_1_1_urovni.png',
     'Уровни фиксации данных о малых архитектурных формах и состояние их обеспеченности', None),
    ('glava1.md', 'Анализ таблицы и литературных источников позволяет констатировать', 'para',
     'risunki/fig_1_2b_pozicionirovanie.png',
     'Положение методов пространственной реконструкции по полноте передачи визуального облика '
     'и производительности визуализации', None),
    ('glava1.md', 'Существенной особенностью базового метода 3DGS', 'para',
     'risunki/fig_1_2a_predstavlenie.png',
     'Представление поверхности объекта полигональной сеткой и набором анизотропных гауссианов', None),
    ('glava1.md', 'Предложенные подходы могут быть разделены на несколько семейств', 'para',
     'risunki/fig_1_4a_semeystva.png',
     'Семейства методов выделения отдельных объектов из 3DGS-сцены', None),
    ('glava1.md', 'Данное различие не является формальным', 'para',
     'risunki/fig_1_4b_vybros_obb.png',
     'Влияние выбросного гауссиана на двумерные метрики качества и на габаритный размер объекта', None),

    ('glava2.md', 'Устойчивость к отражающим и глянцевым поверхностям', 'para',
     'risunki/foto/foto_2_1_1_bliki_planetariy.jpg',
     'Воспроизведение бликов на глянцевых поверхностях в 3DGS-модели', None),
    ('glava2.md', 'Инвариантность к слаботекстурированным поверхностям', 'para',
     'risunki/foto/foto_2_1_3_slozhnye_poverhnosti.jpg',
     'Характерные элементы МАФ, затрудняющие фотограмметрическую реконструкцию', None),
    ('glava2.md', 'Выбранные три типа объектов обеспечивают покрытие', 'para',
     'risunki/foto/foto_2_1_2_klassy_slozhnosti.jpg',
     'Объекты экспериментальной апробации трёх уровней геометрической сложности', None),
    ('glava2.md', 'LichtFeld Studio отвечает всем сформулированным критериям', 'para',
     'risunki/foto/skrin_2_3_lichtfeld_interfeys.jpg',
     'Интерфейс LichtFeld Studio с загруженным набором данных: разрежённое облако точек, '
     'положения камер и параметры обучения', None),
    ('glava2.md', 'Траектория движения камеры.', 'para',
     'risunki/fig_2_4_2_semka.png',
     'Схема съёмки объекта: полный обход, высотные уровни и характерная ошибка неполного обхода', None),
    ('glava2.md', 'COLMAP (Structure from Motion) является стандартным инструментом', 'para',
     'risunki/foto/skrin_2_4_5_colmap_registraciya.jpg',
     'Последовательная регистрация изображений при оценке поз камер в COLMAP '
     '(объект — скамейка с урной)', None),
    ('glava2.md', 'Для видеоданных, извлеченных с равномерным шагом', 'para',
     'risunki/foto/foto_2_4_5_traektorii_kamery.jpg',
     'Результат оценки поз камер: разрежённое облако точек и траектория камеры, вид сверху', None),
    ('glava2.md', 'Указанное обстоятельство определяет порядок следования этапов', 'para',
     'risunki/foto/foto_2_4_6_segmentaciya_3dgs.jpg',
     '3DGS-модель скамейки с урной, обученная без масок и с масками объекта', None),
    ('glava2.md', 'В процессе обучения 3DGS оптимизирует параметры', 'para',
     'risunki/foto/skrin_2_4_7_obuchenie_3dgs.jpg',
     'Ход обучения 3DGS-модели в LichtFeld Studio: уточнение сцены и рост числа гауссианов', None),
    ('glava2.md', 'Опознавание делений шкалы оператором', 'para',
     'risunki/foto/foto_2_5_3_etalon_aruco.jpg',
     'Линейный эталон с кодированными маркерами ArUco и результат их автоматического обнаружения', None),

    ('glava3.md', '| 10 |', 'after',
     'risunki/fig_3_1_1_payplayn.png',
     'Блок-схема сквозного пайплайна обработки данных',
     'Последовательность этапов и их группировка по назначению показаны на рисунке {n}.'),
    ('glava3.md', '#### 3.1.7.', 'after',
     'risunki/foto/skrin_3_1_7_import_dannyh.jpg',
     'Импорт результатов оценки поз камер в LichtFeld Studio',
     'Импорт набора кадров и результатов оценки поз камер выполняется через окно создания проекта '
     '(рисунок {n}).'),
    ('glava3.md', '#### 3.1.11.', 'after',
     'risunki/fig_3_1_11_obb.png',
     'Построение ориентированного ограничивающего параллелепипеда',
     'Схема построения ориентированного ограничивающего параллелепипеда приведена на рисунке {n}.'),
    ('glava3.md', '#### 3.2.5.', 'after',
     'risunki/foto/foto_3_2_5_kachestvo_modeley.jpg',
     '3DGS-модели объектов высокой геометрической сложности (пилотная съёмка)',
     'Пример визуального качества моделей объектов высокой геометрической сложности '
     'приведён на рисунке {n}.'),
    ('glava3.md', '#### 3.2.6.', 'before',
     'risunki/foto/skrin_3_2_5_artefakty_periferii.jpg',
     'Различие качества реконструкции в зоне полного покрытия и на периферии сцены',
     'Качество реконструкции снижается к периферии сцены, покрытой меньшим числом ракурсов '
     '(рисунок {n}).'),
    ('glava3.md', 'Схема (диаграмма процесса)', 'after',
     'risunki/fig_3_5_org_model.png',
     'Организационная модель применения методики в эксплуатирующей организации',
     'Организационный цикл учёта одного объекта представлен на рисунке {n}.'),
]

FILES = ['glava1.md', 'glava2.md', 'glava3.md']


def norm(s):
    return s.strip().lstrip('*').strip()


def find_line(lines, prefix):
    hits = [i for i, l in enumerate(lines) if norm(l).startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f'якорь «{prefix}»: найдено {len(hits)} совпадений, нужно ровно одно')
    return hits[0]


def add_ref(line, n, eol):
    body = line[:-len(eol)] if eol and line.endswith(eol) else line
    stripped = body.rstrip()
    tail = body[len(stripped):]
    ref = f' (рисунок {n})'
    m = re.search(r'\.(\s*\[[\d,\s–-]+\])$', stripped)
    if m:  # «…съемке.[33]» → «…съемке (рисунок N).[33]»
        stripped = stripped[:m.start()] + ref + '.' + m.group(1)
    elif stripped.endswith('.'):
        stripped = stripped[:-1] + ref + '.'
    else:
        stripped = stripped + ref
    return stripped + tail + eol


REF_RE = re.compile(r' \(рисунок \d+\)')
# фраза-ссылка проверяется после снятия «(рисунок N)», поэтому шаблон строится так же
PHRASES = [re.escape(REF_RE.sub('', p[5].replace('{n}', '0'))).replace('0', r'\d+')
           for p in FIGS if p[5]]


def strip_figures(lines):
    """Снять прежние вставки: блоки рисунков, фразы-ссылки, «(рисунок N)»."""
    out, i = [], 0
    while i < len(lines):
        body = lines[i].rstrip('\r\n')
        if body.startswith(MARK):
            # блок: маркер, картинка, пустая, подпись, пустая
            j = i + 1
            while j < len(lines) and not lines[j].startswith('Рисунок '):
                j += 1
            j += 1
            if j < len(lines) and not lines[j].strip():
                j += 1
            # пустая строка перед маркером была добавлена вставкой
            if out and not out[-1].strip():
                out.pop()
            # фраза-ссылка перед блоком
            if out and any(re.fullmatch(ph, out[-1].strip()) for ph in PHRASES):
                out.pop()
                if out and not out[-1].strip():
                    out.pop()
            i = j
            continue
        out.append(REF_RE.sub('', lines[i]))
        i += 1
    return out


def main():
    texts, eols = {}, {}
    for f in FILES:
        with open(ROOT / f, encoding='utf-8', newline='') as fh:
            raw = fh.read()
        eols[f] = '\r\n' if '\r\n' in raw else '\n'
        lines = raw.splitlines(keepends=True)
        if MARK in raw:
            n0 = raw.count(MARK)
            lines = strip_figures(lines)
            print(f'{f}: снято прежних вставок: {n0}')
        texts[f] = lines

    # порядок нумерации — по главе, затем по положению якоря в тексте
    located = []
    for spec in FIGS:
        f = spec[0]
        located.append((FILES.index(f), find_line(texts[f], spec[1]), spec))
    located.sort(key=lambda x: (x[0], x[1]))

    numbered = [(n, idx, spec) for n, (_, idx, spec) in enumerate(located, start=1)]

    # вставляем снизу вверх, чтобы индексы строк не смещались
    for n, idx, spec in sorted(numbered, key=lambda x: (x[2][0], -x[1])):
        f, prefix, mode, img, caption, phrase = spec
        lines, eol = texts[f], eols[f]
        if not lines[idx].endswith(('\n', '\r')):
            lines[idx] += eol
        block = [eol]
        if mode == 'para':
            lines[idx] = add_ref(lines[idx], n, eol)
        else:
            block += [phrase.format(n=n) + eol, eol]
        if mode == 'before':
            block = block[1:]  # вставка перед заголовком: пустая строка в конце уже есть
        block += [
            f'{MARK}{n} -->' + eol,
            f'![]({img})' + eol,
            eol,
            f'Рисунок {n} — {caption}' + eol,
            eol,
        ]
        pos = idx if mode == 'before' else idx + 1
        lines[pos:pos] = block
        print(f'{f}: рисунок {n:>2} после строки {idx + 1}: {caption[:60]}')

    for f in FILES:
        with open(ROOT / f, 'w', encoding='utf-8', newline='') as fh:
            fh.write(''.join(texts[f]))
    print('готово')


if __name__ == '__main__':
    main()
