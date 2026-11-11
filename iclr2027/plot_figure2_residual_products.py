#!/usr/bin/env python3
"""Recreate Figure 2 from the values printed in its source Matplotlib PDFs.

The input PDFs are the authoritative source: matrices are transcribed from their
embedded numeric annotations, while the heatmap palette and panel layout follow
the supplied plots.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

OUT = Path(__file__).parent / "figures"

MHC = [
    np.array([[.91, .03, .03, .03], [.06, .94, .00, .00], [.03, .00, .97, .00], [.00, .00, .00, 1.00]]),
    np.array([[1.00, .00, .00, .00], [.00, .90, .10, .00], [.00, .01, .91, .09], [.00, .09, .00, .91]]),
    np.array([[1.00, .00, .00, .00], [.00, 1.00, .00, .00], [.00, .05, .95, .00], [.00, .00, .05, .95]]),
    np.array([[.29, .25, .23, .22], [.24, .26, .30, .21], [.12, .29, .35, .24], [.17, .29, .29, .25]]),
]
ITO = [
    np.array([[1.00, -.00, .01, -.00], [.01, 1.00, .01, .00], [-.00, -.00, 1.00, -.00], [-.00, .00, -.00, 1.00]]),
    np.array([[1.00, -.03, -.01, -.07], [-.06, 1.00, -.01, -.06], [.01, .00, 1.00, -.00], [-.05, -.02, -.01, 1.00]]),
    np.array([[1.00, .01, -.00, -.01], [.00, 1.00, -.00, -.01], [.00, .01, 1.00, -.01], [.00, .01, -.00, 1.00]]),
    np.array([[1.15, -.33, -.06, -.56], [.08, .90, .31, -.24], [-.30, -.10, 1.07, -.18], [-.42, -.04, -.25, 1.14]]),
]

def draw(matrices, *, cmap, vmin, vmax, out_path):
    # Ratios leave a clean symbolic gap for H_47 x ... x H_1 and H_1 x H_0.
    fig = plt.figure(figsize=(10.6, 2.82), facecolor="white")
    gs = GridSpec(1, 8, figure=fig,
                  width_ratios=[1.0, .52, 1.0, .42, 1.0, .32, 1.0, .34],
                  left=.045, right=.93, bottom=.16, top=.89, wspace=0.0)
    panel_cols = [0, 2, 4, 6]
    symbols = [fig.add_subplot(gs[0, i]) for i in (1, 3, 5)]
    cax = fig.add_subplot(gs[0, 7])
    panel_titles = [r"$H_{47}$", r"$H_1$", r"$H_0$", r"$G$"]
    images = []
    for k, col in enumerate(panel_cols):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(matrices[k], cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest", aspect="equal")
        images.append(im)
        ax.set_title(panel_titles[k], fontsize=12, pad=3)
        ax.set_xticks(range(4), labels=range(4), fontsize=8)
        ax.set_yticks(range(4), labels=range(4), fontsize=8)
        ax.set_xlabel("out", fontsize=8, labelpad=1)
        if k == 0:
            ax.set_ylabel("in", fontsize=8, labelpad=2)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="both", direction="out", length=2, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(.7)
        for i in range(4):
            for j in range(4):
                value = matrices[k][i, j]
                label = f"{value:.2f}"
                # Contrast labels against each cell while retaining the original palette.
                rgba = im.cmap(im.norm(value))
                luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                ax.text(j, i, label, ha="center", va="center", fontsize=6.5,
                        color="white" if luminance < .48 else "black")
    for ax in symbols:
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    symbols[0].text(.5, .5, r"$\times\;\cdots\;\times$", ha="center", va="center", fontsize=13)
    symbols[1].text(.5, .5, r"$\times$", ha="center", va="center", fontsize=13)
    symbols[2].text(.5, .5, r"$\times$", ha="center", va="center", fontsize=13)
    cb = fig.colorbar(images[-1], cax=cax)
    if cmap == "coolwarm":
        cb.set_ticks([-1.0, -.75, -.5, -.25, 0.0, .25, .5, .75, 1.0])
    cb.set_label("entry", fontsize=8, labelpad=2)
    cb.ax.tick_params(labelsize=8, length=2, pad=2)
    fig.savefig(out_path, bbox_inches=None, metadata={"Title": "Figure 2 residual-mixer products"})
    plt.close(fig)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    draw(MHC, cmap="viridis", vmin=0.0, vmax=1.0,
         out_path=OUT / "dynamic_mhc_h_res_heatmaps.pdf")
    draw(ITO, cmap="coolwarm", vmin=-1.2, vmax=1.2,
         out_path=OUT / "ito_h_res_heatmaps.pdf")
