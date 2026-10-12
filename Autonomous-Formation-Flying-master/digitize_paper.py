"""Digitise the paper's Figs. 9 and 10 (position / attitude error vs time) so our results can be
plotted against them. The figures are raster images inside the PDF; the blue curve is traced
pixel-column by pixel-column and mapped to data units with the gridline positions below.

Usage:  python digitize_paper.py ../Punnoose.pdf      (needs: pip install pypdf pillow)
Output: paper_data/paper_fig9.csv, paper_data/paper_fig10.csv   (t, pos_err_m, att_err_deg)
Accuracy: about +-1 px, i.e. ~0.1 m / ~0.02 deg for Fig. 9 and ~0.2 m / ~0.02 deg for Fig. 10.
"""
import io, sys
import numpy as np
from PIL import Image
from pypdf import PdfReader
from config import ROOT

OUT_DIR = ROOT / "paper_data"

# image name in the PDF (page 4) -> calibration, read off the gridlines of each image:
#   x: pixel column of t=0 and t=10;  panel: (row range, (row, value), (row, value))
FIGS = {
    "fig9":  dict(image="Im9.png",  x=(108, 949),
                  pos=((40, 300), (254, 6.0), (67, 18.0)), att=((330, 640), (614, 0.0), (381, 3.0))),
    "fig10": dict(image="Im10.png", x=(81, 711),
                  pos=((40, 240), (227, 0.0), (76, 30.0)), att=((280, 490), (476, 0.0), (359, 2.0))),
}
DT = 0.1


def blue_mask(rgb):
    r, g, b = (rgb[..., k].astype(int) for k in range(3))
    return (b > 200) & (r < 120) & (g > 100) & (g < 200)     # Julia Plots default blue (#009AFA)


def trace(mask, rows, x0, x1, cal0, cal1, n):
    """Value of the curve at each sample t = 0, DT, ... (n samples)."""
    sub = mask[rows[0]:rows[1]]
    to_val = lambda row: cal0[1] + (row + rows[0] - cal0[0]) * (cal1[1] - cal0[1]) / (cal1[0] - cal0[0])
    px_per_s = (x1 - x0) / 10.0
    span = {}
    for c in range(sub.shape[1]):
        r = np.where(sub[:, c])[0]
        if len(r):
            span[c] = (r.min(), r.max())
    centre = lambda c: np.mean(span[c]) if c in span else np.nan
    vals = []
    for k in range(n):
        c = int(round(x0 + k * DT * px_per_s))
        cs = [cc for cc in range(c - 1, c + 2) if cc in span]
        if not cs:
            vals.append(np.nan); continue
        top, bot = min(span[cc][0] for cc in cs), max(span[cc][1] for cc in cs)
        left, right = centre(c - 3), centre(c + 3)
        neigh = [v for v in (left, right) if not np.isnan(v)]
        if neigh and top < min(neigh) - 1:          # local peak (rows grow downwards)
            row = top + 1
        elif neigh and bot > max(neigh) + 1:        # local valley
            row = bot - 1
        else:
            row = (top + bot) / 2
        vals.append(to_val(row))
    return np.array(vals)


def main(pdf):
    page = PdfReader(pdf).pages[3]
    images = {im.name: im.data for im in page.images}
    OUT_DIR.mkdir(exist_ok=True)
    for name, f in FIGS.items():
        rgb = np.asarray(Image.open(io.BytesIO(images[f["image"]])).convert("RGB"))
        mask = blue_mask(rgb)
        last_col = np.where(mask.any(0))[0].max()
        n = int(round((last_col - f["x"][0]) / ((f["x"][1] - f["x"][0]) / 10.0) / DT)) + 1
        t = np.arange(n) * DT
        pos = trace(mask, f["pos"][0], *f["x"], f["pos"][1], f["pos"][2], n)
        att = trace(mask, f["att"][0], *f["x"], f["att"][1], f["att"][2], n)
        np.savetxt(OUT_DIR / f"paper_{name}.csv", np.column_stack([t, pos, att]), delimiter=",",
                   fmt="%.3f", header="t_s,pos_err_m,att_err_deg", comments="")
        print(f"paper {name}: {n} samples  mean position error {np.nanmean(pos):.2f} m  "
              f"mean attitude error {np.nanmean(att):.2f} deg")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "Punnoose.pdf"))
