"""Camera 4 exact-fit timestamp box re-render (v2).

Problem: cam4_final4.mp4 draws a fixed 431x63 black box behind overlay text that is
at most ~223px wide -> ~200px of dead black space in every frame. User rejected it.

Fix: shrink box to x 0..235 and restore the freed region (x 236..430, y 0..62):
  - y 0..30 : true scene from cam4_nobox.mp4 with the (wider, boxless) timestamp
              text inpainted via local median; saturated annotation pixels
              (cyan EMP-01 labels/boxes) are preserved.
  - y 31..62: vertical gradient (top = median of cleaned band above,
              bottom = median of rows 64..72 below) + light noise, then
              saturated annotation pixels from cam4_final4 are pasted back on top
              (person boxes/labels that overlap this corner).
Overlay text inside the new box is never touched.
"""
import subprocess
import numpy as np
from PIL import Image, ImageFilter

W, H = 960, 540
NFRAMES = 4491
FPS = '3693/125'
NEW_X1 = 235          # new box right edge (inclusive)
OLD_X1 = 430
XS = slice(NEW_X1 + 1, OLD_X1 + 1)   # 236..430

def reader(path):
    return subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-i', path,
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, bufsize=10**8)

def writer(path):
    return subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{W}x{H}', '-r', FPS, '-i', '-',
         '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
         '-pix_fmt', 'yuv420p', '-r', FPS, path],
        stdin=subprocess.PIPE, bufsize=10**8)

def main():
    rng = np.random.default_rng(7)
    r_final = reader('cam4_final4.mp4')
    r_nobox = reader('cam4_nobox.mp4')
    w = writer('cam4_exactfit.mp4')
    fsz = W * H * 3
    for i in range(NFRAMES):
        f4 = np.frombuffer(r_final.stdout.read(fsz), dtype=np.uint8).reshape(H, W, 3).copy()
        nb = np.frombuffer(r_nobox.stdout.read(fsz), dtype=np.uint8).reshape(H, W, 3)

        # ---- 1) rows 0..30: scene from nobox, white timestamp text inpainted ----
        # nobox draws the timestamp in a wider font with no box; erase it fully
        # (aggressive threshold + dilation) while keeping saturated annotations
        band = nb[0:31, XS].astype(int)
        mx = band.max(axis=2); mn = band.min(axis=2)
        m = (mx > 25) & ((mx - mn) < 80)   # white/gray text, not cyan labels
        m = np.array(Image.fromarray(m).filter(ImageFilter.MaxFilter(5)))
        clean = band.copy()
        if m.any():
            bg = np.median(band[~m.astype(bool)].reshape(-1, 3), axis=0)
            clean[m.astype(bool)] = bg
        top_c = np.median(clean.reshape(-1, 3), axis=0)
        f4[0:31, XS] = clean.astype(np.uint8)

        # ---- 2) rows 31..62: gradient + noise, then annotation pixels on top ----
        orig = f4[31:63, XS].astype(int)   # original: black box + annotations
        below = f4[64:73, XS].reshape(-1, 3).astype(int)
        bot_c = np.median(below, axis=0)
        t = np.linspace(0, 1, 32)[:, None, None]
        grad = top_c[None, None, :] * (1 - t) + bot_c[None, None, :] * t
        grad = np.repeat(grad, OLD_X1 - NEW_X1, axis=1)
        grad += rng.normal(0, 2, grad.shape)
        grad = np.clip(grad, 0, 255).astype(np.uint8)
        # annotation pixels: anything not near-black in the original box region
        # (cyan boxes/labels, green/red/white text) - the box itself is ~0
        omx = orig.max(axis=2)
        annot = omx > 40
        grad[annot] = orig[annot]
        f4[31:63, XS] = grad

        w.stdin.write(f4.tobytes())
        if (i + 1) % 900 == 0:
            print(f'{i + 1}/{NFRAMES}', flush=True)

    w.stdin.close(); w.wait()
    r_final.wait(); r_nobox.wait()
    print('done')

if __name__ == '__main__':
    main()
