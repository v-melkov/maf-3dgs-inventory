"""Вид сверху на результат оценки поз камер (COLMAP): разрежённое облако
точек и траектория камеры. Читает images.bin / points3D.bin напрямую.

python colmap_topview.py <sparse-каталог> <out.png> "<заголовок>" [радиус_обрезки]
"""
import struct, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_images(path):
    out = []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            iid, = struct.unpack('<i', f.read(4))
            q = np.array(struct.unpack('<4d', f.read(32)))
            t = np.array(struct.unpack('<3d', f.read(24)))
            struct.unpack('<i', f.read(4))
            name = b''
            while (c := f.read(1)) != b'\x00':
                name += c
            n2d, = struct.unpack('<Q', f.read(8))
            f.seek(24 * n2d, 1)
            out.append((name.decode(), q, t))
    return out


def read_points(path):
    xyz, rgb, err = [], [], []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            f.read(8)
            xyz.append(struct.unpack('<3d', f.read(24)))
            rgb.append(struct.unpack('<3B', f.read(3)))
            err.append(struct.unpack('<d', f.read(8))[0])
            tl, = struct.unpack('<Q', f.read(8))
            f.seek(8 * tl, 1)
    return np.array(xyz), np.array(rgb) / 255.0, np.array(err)


def qmat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def main(sp, out, title, rclip=2.2):
    ims = sorted(read_images(f'{sp}/images.bin'), key=lambda r: r[0])
    R = [qmat(q) for _, q, _ in ims]
    C = np.array([-r.T @ t for r, (_, _, t) in zip(R, ims)])
    look = np.array([r.T @ [0, 0, 1] for r in R])
    up = -np.mean([r.T @ [0, 1, 0] for r in R], axis=0); up /= np.linalg.norm(up)
    P, col, err = read_points(f'{sp}/points3D.bin')
    # базис горизонтальной плоскости
    a = np.cross(up, [1, 0, 0]);  a = a if np.linalg.norm(a) > .1 else np.cross(up, [0, 1, 0])
    e1 = a / np.linalg.norm(a); e2 = np.cross(up, e1)
    ctr = np.median(P, axis=0)
    # центр объекта — медиана точек, на которые смотрят камеры: ближайшая точка лучей
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c, d in zip(C, look):
        M = np.eye(3) - np.outer(d, d); A += M; b += M @ c
    ctr = np.linalg.lstsq(A, b, rcond=None)[0]
    pc = lambda X: np.c_[(X - ctr) @ e1, (X - ctr) @ e2]
    c2, p2 = pc(C), pc(P)
    rcam = np.median(np.linalg.norm(c2, axis=1))
    keep = (np.linalg.norm(p2, axis=1) < rclip * rcam) & (err < np.percentile(err, 95))
    h = (P - ctr) @ up
    keep &= np.abs(h - np.median(h[keep])) < 3 * rcam
    l2 = pc(C + look) - c2; l2 /= np.linalg.norm(l2, axis=1, keepdims=True) + 1e-9

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=220)
    ax.scatter(p2[keep, 0], p2[keep, 1], s=0.6, c=col[keep], linewidths=0, alpha=.9, rasterized=True)
    ax.plot(c2[:, 0], c2[:, 1], '-', color='#7a7a7a', lw=.7, zorder=2)
    sc = ax.scatter(c2[:, 0], c2[:, 1], c=np.arange(len(c2)), cmap='viridis', s=14,
                    edgecolors='white', linewidths=.4, zorder=3)
    s = rcam * .16
    for (x, y), (u, v) in zip(c2[::2], l2[::2]):
        ax.plot([x, x + u * s], [y, y + v * s], color='#1a1a1a', lw=.6, zorder=2)
    ax.plot(*c2[0], marker='o', ms=9, mfc='none', mec='#a52a2a', mew=1.6, zorder=4)
    ax.annotate('начало', c2[0], xytext=(8, 8), textcoords='offset points', fontsize=9, color='#a52a2a')
    ax.set_aspect('equal'); lim = rclip * rcam * 1.02
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values(): sp_.set_color('#9a9a9a')
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(sc, ax=ax, fraction=.035, pad=.02); cb.set_label('порядковый номер кадра', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.text(.01, .01, f'кадров: {len(C)}   точек: {keep.sum():,}'.replace(',', ' '),
            transform=ax.transAxes, fontsize=8, color='#333', va='bottom')
    fig.tight_layout(); fig.savefig(out, facecolor='white'); print(out, len(C), keep.sum())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 2.2)
