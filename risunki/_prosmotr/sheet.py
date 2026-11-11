import sys, os, glob
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
pat = sys.argv[4] if len(sys.argv) > 4 else '*'
files = sorted(f for f in glob.glob(os.path.join(src, pat)) if f.lower().endswith(('.jpg','.jpeg','.png')))
if not files: sys.exit(f'нет файлов {src}')
step = max(1, len(files) // n)
pick = files[::step][:n]
tw = 360
thumbs = []
for f in pick:
    im = Image.open(f); im.draft('RGB', (tw*2, tw*2))
    if im.mode not in ('RGB','L'):
        try: im = im.convert('RGB')
        except Exception: im = Image.eval(im, lambda v: v >> 8).convert('RGB')
    im.thumbnail((tw, tw)); thumbs.append((os.path.basename(f), im.convert('RGB')))
cols = 6; rows = (len(thumbs)+cols-1)//cols
W, H = cols*(tw+6), rows*(tw+28)
sheet = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(sheet)
for k,(name,im) in enumerate(thumbs):
    x, y = (k%cols)*(tw+6), (k//cols)*(tw+28)
    sheet.paste(im, (x + (tw-im.width)//2, y + (tw-im.height)//2))
    d.text((x+4, y+tw+6), name[:48], fill=(0,0,0))
sheet.save(out, quality=80); print(out, len(files), 'files, shown', len(thumbs))
