#!/usr/bin/env python3
"""
a3_to_a4.py - Convertit une partition PDF A3 en PDF A4 optimisé.

Modes :
  reflow : détecte les systèmes (lignes de portées), rogne les marges et les
           re-répartit sur des pages A4 (défaut).
  split  : A3 paysage = 2 pages A4 côte à côte -> coupe en deux, sans perte de taille.
  fit    : rogne les marges de chaque page et la réduit simplement sur une A4.
  scan   : pour les SCANS de livrets A3 (2 pages A4 par feuille) : coupe au pli,
           supprime les pages blanches, redresse, nettoie le fond gris, rogne les
           marges, remet les pages dans l'ordre voulu (--order) et sort des A4.

Le rendu reste vectoriel (PyMuPDF réutilise le contenu d'origine, pas de rastérisation).

Usage :
  python a3_to_a4.py entree.pdf sortie.pdf --mode reflow
  python a3_to_a4.py            (sans argument : ouvre la petite interface graphique)
"""
import argparse
import io
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter

try:
    import pymupdf as fitz  # PyMuPDF (nom récent)
except ImportError:
    import fitz  # anciens PyMuPDF
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

MM = 72 / 25.4
A4_W, A4_H = 210 * MM, 297 * MM


# ----------------------------------------------------------------- détection
def ink_mask(page, dpi=100, thr=200):
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return a < thr, dpi / 72.0


def content_bbox(page, pad_mm=1.5):
    mask, k = ink_mask(page)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return page.rect
    p = pad_mm * MM
    r = fitz.Rect(xs.min() / k - p, ys.min() / k - p, (xs.max() + 1) / k + p, (ys.max() + 1) / k + p)
    return r & page.rect


def find_systems(page, gap_mm=12.0, pad_mm=1.5, noise=0.002):
    """Retourne la liste des bandes verticales (y0, y1) en points + l'étendue x globale."""
    mask, k = ink_mask(page)
    rows = mask.mean(axis=1) > noise
    runs, start = [], None
    for i, v in enumerate(rows):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(rows)])
    if not runs:
        return [], None

    gap_px = gap_mm * MM * k
    merged = [runs[0]]
    for r in runs[1:]:
        if r[0] - merged[-1][1] < gap_px:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    cols = np.where(mask.sum(axis=0) >= 2)[0]
    p = pad_mm * MM
    x0 = max(page.rect.x0, cols.min() / k - p)
    x1 = min(page.rect.x1, (cols.max() + 1) / k + p)
    bands = [(max(page.rect.y0, a / k - p), min(page.rect.y1, b / k + p)) for a, b in merged]
    return bands, (x0, x1)


# -------------------------------------------------------------------- modes
def mode_reflow(src, out, margin_mm, spacing_mm, gap_mm, justify):
    margin = margin_mm * MM
    spacing = spacing_mm * MM
    systems = []  # (page_no, Rect)
    for pno, page in enumerate(src):
        bands, xr = find_systems(page, gap_mm=gap_mm)
        for y0, y1 in bands:
            systems.append((pno, fitz.Rect(xr[0], y0, xr[1], y1)))
    if not systems:
        sys.exit("Aucun contenu détecté.")

    avail_w = A4_W - 2 * margin
    avail_h = A4_H - 2 * margin
    scale = avail_w / max(r.width for _, r in systems)

    # 1) répartition des systèmes par page
    pages, cur, used = [], [], 0.0
    for pno, r in systems:
        h = r.height * scale
        extra = spacing if cur else 0
        if cur and used + extra + h > avail_h:
            pages.append(cur)
            cur, used, extra = [], 0.0, 0
        cur.append((pno, r, h))
        used += extra + h
    pages.append(cur)

    # 2) dessin
    for idx, items in enumerate(pages):
        page = out.new_page(width=A4_W, height=A4_H)
        total = sum(h for _, _, h in items)
        gap = spacing
        if justify and idx < len(pages) - 1 and len(items) > 1:
            gap = min((avail_h - total) / (len(items) - 1), spacing * 3)
        y = margin
        for pno, r, h in items:
            dest = fitz.Rect(margin, y, margin + r.width * scale, y + h)
            page.show_pdf_page(dest, src, pno, clip=r)
            y += h + gap


def mode_split(src, out, margin_mm):
    m = margin_mm * MM
    target = fitz.Rect(m, m, A4_W - m, A4_H - m)
    for pno, page in enumerate(src):
        r = page.rect
        if r.width > r.height:  # paysage : 2 colonnes
            halves = [fitz.Rect(r.x0, r.y0, r.x0 + r.width / 2, r.y1),
                      fitz.Rect(r.x0 + r.width / 2, r.y0, r.x1, r.y1)]
        else:  # portrait : 2 lignes
            halves = [fitz.Rect(r.x0, r.y0, r.x1, r.y0 + r.height / 2),
                      fitz.Rect(r.x0, r.y0 + r.height / 2, r.x1, r.y1)]
        for clip in halves:
            out.new_page(width=A4_W, height=A4_H).show_pdf_page(target, src, pno, clip=clip)


def mode_fit(src, out, margin_mm):
    m = margin_mm * MM
    target = fitz.Rect(m, m, A4_W - m, A4_H - m)
    for pno, page in enumerate(src):
        clip = content_bbox(page)
        out.new_page(width=A4_W, height=A4_H).show_pdf_page(target, src, pno, clip=clip)


# ---------------------------------------------------------------- mode scan
def _gray(page, dpi, clip=None):
    pix = page.get_pixmap(dpi=dpi, clip=clip, colorspace=fitz.csGRAY)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    return Image.fromarray(a.copy(), "L")


def find_gutter(page):
    """Abscisse (pt) du pli : colonne blanche la plus proche du centre."""
    k = 40 / 72.0
    a = np.asarray(_gray(page, 40))
    cols = (a < 150).sum(axis=0).astype(float)
    w = len(cols)
    lo, hi = int(w * 0.4), int(w * 0.6)
    seg = cols[lo:hi]
    best = np.where(seg <= seg.min() + 0.002 * a.shape[0])[0] + lo
    x = best[np.argmin(np.abs(best - w / 2))]
    return (x + 0.5) / k


def estimate_skew(ink, max_deg=3.0):
    """ink : image 'L' (encre = 255). Angle (degrés) qui rend les lignes de portée horizontales."""
    def score(angle):
        r = ink.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        return float(np.asarray(r, dtype=np.float32).sum(axis=1).var())
    coarse = np.arange(-max_deg, max_deg + 1e-6, 0.25)
    a0 = max(coarse, key=score)
    fine = np.arange(a0 - 0.25, a0 + 0.25 + 1e-6, 0.05)
    return float(max(fine, key=score))


def _bbox_px(ink_arr, noise=3):
    rows = np.where(ink_arr.sum(axis=1) >= noise * 255)[0]
    cols = np.where(ink_arr.sum(axis=0) >= noise * 255)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1


def flatten_background(img):
    """Supprime l'ombre du pli et le fond grisâtre, garde le noir."""
    w, h = img.size
    f = 8
    small = img.resize((max(1, w // f), max(1, h // f)), Image.BILINEAR)
    bg = small.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(5))
    bg = np.asarray(bg.resize((w, h), Image.BILINEAR), dtype=np.float32).clip(1, 255)
    a = np.asarray(img, dtype=np.float32) / bg * 255
    a = ((a - 70) / (225 - 70)).clip(0, 1) * 255
    return Image.fromarray(a.astype(np.uint8), "L")


def analyze_scan(src, blank_thr, deskew, light=False, split=True):
    """Découpe chaque feuille A3 en pages A4 et les analyse (blanche ? angle ? zone utile ?)."""
    halves = []  # dict par demi-feuille, numérotées à partir de 1
    for pno, page in enumerate(src):
        r = page.rect
        if split and r.width > r.height:
            gx = find_gutter(page)
            clips = [fitz.Rect(r.x0, r.y0, gx, r.y1), fitz.Rect(gx, r.y0, r.x1, r.y1)]
        else:
            clips = [r]
        for clip in clips:
            img = _gray(page, 100, clip)
            ink = Image.fromarray(((np.asarray(img) < 150) * 255).astype(np.uint8), "L")
            frac = (np.asarray(ink) > 0).mean()
            h = dict(n=len(halves) + 1, pno=pno, clip=clip, frac=frac, blank=frac < blank_thr,
                     angle=0.0, thumb=img)
            if not h["blank"] and not light:
                if deskew:
                    h["angle"] = estimate_skew(ink.resize((ink.width // 2, ink.height // 2), Image.BILINEAR))
                    if abs(h["angle"]) > 0.03:
                        ink = ink.rotate(h["angle"], resample=Image.BILINEAR, fillcolor=0)
                bb = _bbox_px(np.asarray(ink))
                if bb is None:
                    h["blank"] = True
                else:
                    p = 2 * MM * 100 / 72  # 2 mm de marge autour du contenu
                    x0, y0, x1, y1 = bb
                    h["bbox"] = (max(0, x0 - p), max(0, y0 - p), min(ink.width, x1 + p), min(ink.height, y1 + p))
            halves.append(h)
    return halves


# ------------------------------------------- lecture des numéros de page (OCR)
_OCR = {"tried": False, "name": None, "fn": None, "tess": None, "error": None}


def _ocr_vision(img, charset=None):
    """OCR intégré à macOS (framework Vision, via pyobjc)."""
    import Vision
    from Foundation import NSData
    buf = io.BytesIO()
    img.save(buf, "PNG")
    raw = buf.getvalue()
    data = NSData.dataWithBytes_length_(raw, len(raw))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    res = handler.performRequests_error_([req], None)
    ok = res[0] if isinstance(res, tuple) else res
    if not ok:
        raise RuntimeError("Vision n'a pas pu lire l'image")
    items = []
    for r in (req.results() or []):
        cands = r.topCandidates_(1)
        if not cands:
            continue
        bb = r.boundingBox()  # normalisé, origine en bas à gauche
        items.append(dict(text=str(cands[0].string()),
                          x=bb.origin.x + bb.size.width / 2,
                          y=1 - (bb.origin.y + bb.size.height / 2)))
    return items


def _find_tesseract():
    base = getattr(sys, "_MEIPASS", None)  # application empaquetée : Tesseract est embarqué
    if base:
        exe = os.path.join(base, "tesseract", "tesseract.exe" if sys.platform.startswith("win") else "tesseract")
        if os.path.exists(exe):
            os.environ["TESSDATA_PREFIX"] = os.path.join(base, "tesseract", "tessdata")
            return exe
    t = shutil.which("tesseract")
    if t:
        return t
    for c in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract"):
        if os.path.exists(c):
            return c
    return None


def _ocr_tesseract(img, charset=None):
    """charset : None (texte libre), 'digits' ou 'letters' (un seul mot/caractère attendu)."""
    white = {"digits": "0123456789", "letters": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}.get(charset)
    W, H = img.size
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bande.png")
        img.save(path)
        for psm in (("7", "8", "10") if white else ("11",)):
            cmd = [_OCR["tess"], path, "stdout", "--psm", psm]
            if white:
                cmd += ["-c", "tessedit_char_whitelist=" + white]
            out = subprocess.run(cmd + ["tsv"], capture_output=True, text=True, timeout=90,
                                 stdin=subprocess.DEVNULL,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            items = []
            for line in out.splitlines()[1:]:
                c = line.split("\t")
                if len(c) < 12 or c[0] != "5":
                    continue
                txt = c[11].strip()
                try:
                    conf = float(c[10])
                    left, top, w, h = (int(v) for v in c[6:10])
                except ValueError:
                    continue
                if txt and conf >= 30:
                    items.append(dict(text=txt, conf=conf, x=(left + w / 2) / W, y=(top + h / 2) / H))
            if items:
                return items
    return []


def get_ocr(retry=False):
    """Retourne (nom, fonction) du moteur de lecture disponible, ou (None, None)."""
    if retry:
        _OCR["tried"] = False
    if not _OCR["tried"]:
        _OCR.update(tried=True, name=None, fn=None, error=None)
        if sys.platform == "darwin":
            try:
                import Vision  # noqa: F401
                import Foundation  # noqa: F401
                _OCR.update(name="Vision", fn=_ocr_vision)
            except Exception as e:  # noqa
                _OCR["error"] = str(e)
        if _OCR["fn"] is None:
            t = _find_tesseract()
            if t:
                _OCR.update(name="Tesseract", fn=_ocr_tesseract, tess=t)
    return _OCR["name"], _OCR["fn"]


def _as_number(text):
    t = text.strip().strip(".,;:-–—•·()[]|")
    return int(t) if t.isdigit() and len(t) <= 3 and int(t) > 0 else None


def _runs(flags, merge=0):
    """Intervalles [debut, fin[ de True consécutifs ; fusionne ceux séparés de moins de `merge`."""
    runs, start = [], None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(flags)])
    out = []
    for r in runs:
        if out and r[0] - out[-1][1] < merge:
            out[-1][1] = r[1]
        else:
            out.append(r)
    return out


def _edge_number(g, fn, which, dpi):
    """Cherche le numéro de page sur la ligne de texte tout en haut (ou tout en bas) de la page.
    On isole cette ligne, on la découpe en groupes séparés par de grands blancs, et on ne lit
    que les groupes ÉTROITS situés aux extrémités (un numéro) : le titre, la mention d'édition
    ou la musique ne sont jamais lus comme des numéros."""
    mm = dpi / 25.4
    H, W = g.shape
    ink = g < 110
    zone = int(H * 0.20)
    off = 0 if which == "top" else H - zone
    sub = ink[off:off + zone]
    runs = [r for r in _runs(sub.sum(axis=1) >= 4, merge=int(1.0 * mm)) if r[1] - r[0] >= 0.8 * mm]
    if not runs:
        return None
    r0, r1 = runs[0] if which == "top" else runs[-1]
    if (r1 - r0) > 7 * mm:  # trop haut pour une ligne d'en-tête : ce n'est pas un numéro
        return None
    y0, y1 = max(0, off + r0 - 3), min(H, off + r1 + 3)
    groups = _runs(ink[y0:y1].sum(axis=0) >= 1, merge=int(7 * mm))
    groups = [gr for gr in groups if (gr[1] - gr[0]) >= 0.8 * mm]
    if not groups:
        return None
    found = set()
    for gr in {0: groups[0], 1: groups[-1]}.values():
        wmm = (gr[1] - gr[0]) / mm
        if wmm > 14:
            continue
        pad = int(1.5 * mm)
        crop = np.where(ink[y0:y1, max(0, gr[0] - pad):min(W, gr[1] + pad)], 0, 255).astype(np.uint8)
        im = Image.fromarray(crop, "L")
        im = ImageOps.expand(im, border=int(3 * mm), fill=255)
        im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
        text = " ".join(i["text"] for i in fn(im, charset="digits"))
        n = _as_number(text)
        if n:
            found.add(n)
    return found.pop() if len(found) == 1 else None


def detect_page_numbers(src, pages, dpi=300):
    """Ajoute h['num'] (numéro imprimé, ou None) à chaque page. False si aucun OCR dispo."""
    name, fn = get_ocr()
    for h in pages:
        h["num"] = None
    if fn is None:
        return False
    for h in pages:
        g = np.asarray(_gray(src[h["pno"]], dpi, h["clip"]))
        try:
            h["marks_raw"] = detect_marks(g)
        except Exception as e:  # noqa
            _OCR["error"] = str(e)
            h["marks_raw"] = []
        nums = []
        for which in ("top", "bottom"):
            try:
                nums.append(_edge_number(g, fn, which, dpi))
            except Exception as e:  # noqa
                _OCR["error"] = str(e)
                nums.append(None)
        h["num"] = nums[0] if nums[0] is not None else nums[1]
    # repères : on garde le type dominant (chiffres ou lettres) et on résume chaque page
    kinds = Counter(k for h in pages for k, _ in h["marks_raw"])
    dom = kinds.most_common(1)[0][0] if kinds else None
    with_marks = 0
    for h in pages:
        vals = sorted(v for k, v in h["marks_raw"] if k == dom)
        h["mark"], h["mark_range"], h["mark_kind"] = None, None, dom
        if vals:
            h["mark"] = float(statistics.median(vals))
            h["mark_range"] = (vals[0], vals[-1])
            with_marks += 1
    if with_marks < 2:  # trop peu pour être fiable
        for h in pages:
            h["mark"], h["mark_range"] = None, None
    return True


# --------------------------------------- repères de la partition (cercles / carrés)
def _dilate(b):
    o = b.copy()
    o[1:] |= b[:-1]
    o[:-1] |= b[1:]
    o[:, 1:] |= b[:, :-1]
    o[:, :-1] |= b[:, 1:]
    return o


def find_frames(g, dpi=150, rmin_mm=2.6, rmax_mm=5.0):
    """Trouve les cadres de repères : cercles (chiffres, lettres) et carrés (lettres).
    Corrélation FFT avec un anneau/un carré creux de plusieurs tailles. Retourne
    (y, x, rayon, forme) en pixels ; forme 0 = cercle, 1 = carré."""
    mm = dpi / 25.4
    raw = g < 150
    ink = _dilate(raw)
    H, W = ink.shape
    rs = list(range(max(6, int(rmin_mm * mm)), int(rmax_mm * mm) + 1, 2))
    R = rs[-1] + 8
    fh, fw = H + 2 * R, W + 2 * R
    F = np.fft.rfft2(ink.astype(np.float32), s=(fh, fw))
    best = np.zeros((H, W), np.float32)
    barg = np.zeros((H, W), np.int16)
    bshape = np.zeros((H, W), np.int8)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    dists = (np.hypot(yy, xx), np.maximum(np.abs(yy), np.abs(xx)))
    for si, dist in enumerate(dists):
        for r in rs:
            ring = np.abs(dist - r) <= 1.2
            t = np.zeros((fh, fw), np.float32)
            ys, xs = np.nonzero(ring)
            t[(ys - R) % fh, (xs - R) % fw] = 1
            corr = np.fft.irfft2(F * np.conj(np.fft.rfft2(t)), s=(fh, fw))[:H, :W] / ring.sum()
            m = corr > best
            best[m], barg[m], bshape[m] = corr[m], r, si
    ys, xs = np.nonzero(best >= 0.95)
    kept = []
    for i in np.argsort(-best[ys, xs])[:500]:
        y, x = int(ys[i]), int(xs[i])
        r, sq = int(barg[y, x]), int(bshape[y, x])
        if any((y - k[0]) ** 2 + (x - k[1]) ** 2 < (0.9 * max(r, k[2])) ** 2 for k in kept):
            continue
        ri = int(0.7 * r)
        sub = raw[max(0, y - ri):y + ri + 1, max(0, x - ri):x + ri + 1]
        frac = float(sub.mean()) if sub.size else 0.0  # l'intérieur contient le chiffre/la lettre
        if not 0.05 <= frac <= 0.60:
            continue
        if sq:  # les carrés ressemblent à la portée : on exige un cadre isolé sur fond blanc
            ro = r + 6
            y0, x0 = max(0, y - ro), max(0, x - ro)
            sub2 = raw[y0:y + ro + 1, x0:x + ro + 1]
            gy, gx = np.mgrid[y0 - y:y0 - y + sub2.shape[0], x0 - x:x0 - x + sub2.shape[1]]
            dd = np.maximum(np.abs(gy), np.abs(gx))
            ann = (dd >= r + 3) & (dd <= r + 6)
            if best[y, x] < 0.97 or not 0.08 <= frac <= 0.5 or (ann.any() and sub2[ann].mean() > 0.08):
                continue
        kept.append((y, x, r, sq))
    return kept


def _read_mark(im):
    """Lit un repère : ('n', 12) pour un chiffre, ('l', 2) pour la lettre B. None sinon."""
    name, fn = get_ocr()
    if fn is None:
        return None
    if name == "Vision":
        text = "".join(" ".join(i["text"] for i in fn(im)).split())
    else:
        best, text = -1, ""
        for cs in ("digits", "letters"):
            items = fn(im, charset=cs)
            if items:
                conf = sum(i["conf"] for i in items) / len(items)
                if conf > best:
                    best, text = conf, "".join(" ".join(i["text"] for i in items).split())
    if text.isdigit() and 1 <= len(text) <= 3 and int(text) > 0:
        return ("n", int(text))
    if len(text) == 1 and text.isalpha() and text.isupper():
        return ("l", ord(text) - 64)
    return None


def detect_marks(g300):
    """Repères d'une page (ordre de lecture) : liste de ('n'|'l', valeur)."""
    h2, w2 = g300.shape[0] // 2, g300.shape[1] // 2
    g150 = np.asarray(Image.fromarray(g300).resize((w2, h2), Image.BOX))
    out = []
    for y, x, r, sq in sorted(find_frames(g150), key=lambda f: (f[0] // 45, f[1])):
        cy, cx, rr = 2 * y, 2 * x, 2 * r
        half = int(rr * 0.9)
        crop = g300[max(0, cy - half):cy + half + 1, max(0, cx - half):cx + half + 1]
        gy, gx = np.mgrid[-half:-half + crop.shape[0], -half:-half + crop.shape[1]]
        # le cadre lui-même est effacé : on ne garde que ce qu'il contient
        inside = (np.hypot(gy, gx) <= 0.78 * rr) if not sq else (np.maximum(abs(gy), abs(gx)) <= 0.78 * rr)
        ink = (crop < 130) & inside
        ys, xs = np.nonzero(ink)
        if len(ys) < 12:
            continue
        pad = 3
        ink = ink[max(0, ys.min() - pad):ys.max() + pad + 1, max(0, xs.min() - pad):xs.max() + pad + 1]
        im = Image.fromarray(np.where(ink, 0, 255).astype(np.uint8), "L")
        scale = max(1.0, 90.0 / im.height)
        im = ImageOps.expand(im.resize((max(1, int(im.width * scale)), int(im.height * scale)), Image.LANCZOS),
                             border=30, fill=255)
        try:
            res = _read_mark(im)
        except Exception as e:  # noqa
            _OCR["error"] = str(e)
            res = None
        if res:
            out.append(res)
    return out


def mark_label(kind, v):
    return str(v) if kind == "n" else chr(64 + v)


def auto_order(pages):
    """Ordre de lecture (liste des h['n']).
    1. numéros imprimés quand ils existent ; 2. repères (1,2,3… ou A,B,C…) pour placer les pages
    sans numéro, ou pour tout classer si aucun numéro n'est lu ; 3. sinon ordre du scan.
    Une page très claire sans repère ni numéro (couverture) passe en tête."""
    cnt = Counter(p["num"] for p in pages if p.get("num"))
    numbered, unnum = {}, []
    for p in pages:
        if p.get("num") and cnt[p["num"]] == 1:
            numbered[p["num"]] = p
        else:
            unnum.append(p)
    front = []
    if unnum:
        med = statistics.median(p["frac"] for p in pages)
        bare = [p for p in unnum if p.get("mark") is None]
        if bare:
            sparse = min(bare, key=lambda p: p["frac"])
            if sparse["frac"] < 0.3 * med:
                front = [sparse]
                unnum.remove(sparse)
    # pages sans numéro : triées d'après leurs repères (celles sans repère à la suite, ordre du scan)
    unnum.sort(key=lambda p: (p.get("mark") is None, p.get("mark") or 0))
    if not numbered:
        return [p["n"] for p in front + unnum]
    mx = max(numbered)
    slots = [k for k in range(1, mx + 1) if k not in numbered]
    fill = dict(zip(slots, unnum))
    rest = unnum[len(slots):]
    body = [numbered[k] if k in numbered else fill[k] for k in range(1, mx + 1) if k in numbered or k in fill]
    return [p["n"] for p in front + body + rest]


def mode_scan(src, out, margin_mm, order, blank_thr, deskew, clean, out_dpi, preview, split=True):
    m = margin_mm * MM
    # --- passe 1 : découpe + analyse en basse résolution
    halves = analyze_scan(src, blank_thr, deskew, split=split)

    print("Demi-feuilles détectées :")
    for h in halves:
        print(f"  n°{h['n']:2d}  feuille {h['pno'] + 1}  encre {h['frac'] * 100:5.2f} %  "
              f"{'BLANCHE (ignorée)' if h['blank'] else 'angle %+.2f°' % h['angle']}")

    if preview:
        tw = 220
        th = int(tw * halves[0]["thumb"].height / halves[0]["thumb"].width)
        cols = min(6, len(halves))
        rows_ = (len(halves) + cols - 1) // cols
        sheet = Image.new("L", (cols * (tw + 10) + 10, rows_ * (th + 30) + 10), 255)
        d = ImageDraw.Draw(sheet)
        for i, h in enumerate(halves):
            x, y = 10 + (i % cols) * (tw + 10), 10 + (i // cols) * (th + 30)
            t = h["thumb"].resize((tw, th))
            sheet.paste(t, (x, y + 20))
            d.rectangle([x, y + 20, x + tw, y + 20 + th], outline=0)
            d.text((x, y + 4), f"n°{h['n']}" + (" (blanche)" if h["blank"] else ""), fill=0)
        sheet.save(preview)
        print("Aperçu numéroté :", preview)

    pages_ok = [h for h in halves if not h["blank"]]
    if order == "auto":
        order = None
        if detect_page_numbers(src, pages_ok):
            order = auto_order(pages_ok)
            print(f"Numéros de page lus ({get_ocr()[0]}) :")
            for h in pages_ok:
                mk = ""
                if h.get("mark_range"):
                    lo, hi = h["mark_range"]
                    mk = f"  repères {mark_label(h['mark_kind'], lo)} à {mark_label(h['mark_kind'], hi)}"
                print(f"  n°{h['n']:2d} -> {h['num'] if h['num'] else 'numéro non lu'}{mk}")
            print("Ordre retenu (--order) :", ",".join(map(str, order)))
        else:
            print("Lecture automatique des numéros indisponible : ordre du scan conservé.")
    byn = {h["n"]: h for h in halves}
    seq = [byn[n] for n in order] if order else [h for h in halves if not h["blank"]]
    seq = [h for h in seq if not h["blank"]]
    if not seq:
        sys.exit("Aucune page à écrire.")

    # --- échelle commune (médiane), jamais plus grande que ce que la page permet
    kk = 72 / 100.0
    fits = []
    for h in seq:
        x0, y0, x1, y1 = h["bbox"]
        bw, bh = (x1 - x0) * kk, (y1 - y0) * kk
        h["bw"], h["bh"] = bw, bh
        h["land"] = bw > bh  # page paysage -> A4 paysage
        pw, ph = (A4_H, A4_W) if h["land"] else (A4_W, A4_H)
        h["fit"] = min((pw - 2 * m) / bw, (ph - 2 * m) / bh)
        fits.append(h["fit"])
    common = statistics.median(fits)

    # --- passe 2 : haute résolution
    work_dpi = 400
    for h in seq:
        page = src[h["pno"]]
        img = _gray(page, work_dpi, h["clip"])
        if abs(h["angle"]) > 0.03:
            img = img.rotate(h["angle"], resample=Image.BICUBIC, fillcolor=255)
        f = work_dpi / 100.0
        x0, y0, x1, y1 = [int(round(v * f)) for v in h["bbox"]]
        img = img.crop((x0, y0, x1, y1))
        s = min(h["fit"], common)
        tw = max(1, int(round(h["bw"] * s / 72 * out_dpi)))
        th = max(1, int(round(h["bh"] * s / 72 * out_dpi)))
        img = img.resize((tw, th), Image.LANCZOS)
        if clean:
            img = flatten_background(img)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=88, optimize=True)
        w_pt, h_pt = h["bw"] * s, h["bh"] * s
        pw, ph = (A4_H, A4_W) if h["land"] else (A4_W, A4_H)
        dest = fitz.Rect((pw - w_pt) / 2, (ph - h_pt) / 2, (pw + w_pt) / 2, (ph + h_pt) / 2)
        out.new_page(width=pw, height=ph).insert_image(dest, stream=buf.getvalue())


# --------------------------------------------------------------------- main
def convert(inp, outp, mode="reflow", margin=8.0, spacing=6.0, gap=12.0, justify=True,
            order=None, blank_thr=0.002, deskew=True, clean=True, dpi=300, preview=None, split=True):
    src = fitz.open(inp)
    out = fitz.open()
    if mode == "reflow":
        mode_reflow(src, out, margin, spacing, gap, justify)
    elif mode == "split":
        mode_split(src, out, margin)
    elif mode == "scan":
        mode_scan(src, out, margin, order, blank_thr, deskew, clean, dpi, preview, split)
    else:
        mode_fit(src, out, margin)
    out.save(outp, garbage=3, deflate=True)
    return len(out)


def gui():
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("Partition A3 vers A4")
    root.geometry("1000x720")
    root.minsize(780, 600)

    st = {"path": None, "pages": [], "analysed": None, "auto": [], "asked_install": False}
    single = tk.BooleanVar(value=False)
    status = tk.StringVar(value="Commencez par choisir votre partition (un fichier PDF).")
    pad = dict(padx=14, pady=6)

    # ---------- tâches longues en arrière-plan (la fenêtre reste vivante)
    def bg(work, done):
        box = {}

        def target():
            try:
                box["r"] = work()
            except Exception as e:  # noqa
                box["e"] = e

        t = threading.Thread(target=target, daemon=True)
        t.start()

        def poll():
            if t.is_alive():
                root.after(150, poll)
            else:
                done(box.get("r"), box.get("e"))
        poll()

    def set_busy(busy, msg=None):
        if msg:
            status.set(msg)
        state = "disabled" if busy else "normal"
        for b in (btn_choose, btn_create, btn_auto, chk_single):
            b.config(state=state)
        if busy:
            bar.start(12)
        else:
            bar.stop()

    # ---------- bas de fenêtre
    bottom = ttk.Frame(root)
    bottom.pack(side="bottom", fill="x", padx=14, pady=10)
    bar = ttk.Progressbar(bottom, mode="indeterminate")
    bar.pack(fill="x", pady=(0, 6))
    ttk.Label(bottom, textvariable=status, wraplength=740).pack(side="left")
    btn_create = ttk.Button(bottom, text="Créer la partition A4…", command=lambda: create())
    btn_create.pack(side="right")

    # ---------- 1. fichier
    f1 = ttk.LabelFrame(root, text="1. Votre partition (PDF scanné en A3)")
    f1.pack(fill="x", **pad)
    row1 = ttk.Frame(f1)
    row1.pack(fill="x")
    btn_choose = ttk.Button(row1, text="Choisir le PDF…", command=lambda: choose())
    btn_choose.pack(side="left", padx=10, pady=8)
    file_lbl = ttk.Label(row1, text="Aucun fichier choisi")
    file_lbl.pack(side="left", padx=6)
    chk_single = ttk.Checkbutton(f1, variable=single, command=lambda: choose_again(),
                                 text="Mes feuilles ne contiennent qu'une page chacune (ne pas les couper en deux)")
    chk_single.pack(anchor="w", padx=10, pady=(0, 8))

    # ---------- 2. ordre des pages
    f3 = ttk.LabelFrame(root, text="2. Ordre des pages")
    f3.pack(fill="both", expand=True, **pad)
    head = ttk.Frame(f3)
    head.pack(fill="x", padx=10, pady=(6, 2))
    ttk.Label(head, wraplength=720, justify="left",
              text="Les pages blanches sont retirées. Les pages sont classées d'après leurs numéros imprimés "
                   "et les repères de la partition (1, 2, 3… ou A, B, C…). Une page sans aucune indication "
                   "(couverture…) est placée au mieux : vérifiez-la avec les flèches ◀ ▶. "
                   "Double-clic sur une page pour l'agrandir."
              ).pack(side="left")
    btn_auto = ttk.Button(head, text="Reclasser automatiquement", command=lambda: reapply_auto())
    btn_auto.pack(side="right", padx=6)
    wrap = ttk.Frame(f3)
    wrap.pack(fill="both", expand=True, padx=6, pady=6)
    canvas = tk.Canvas(wrap, height=300, highlightthickness=0)
    hbar = ttk.Scrollbar(wrap, orient="horizontal", command=canvas.xview)
    canvas.configure(xscrollcommand=hbar.set)
    canvas.pack(side="top", fill="both", expand=True)
    hbar.pack(side="bottom", fill="x")
    inner = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def zoom(p):
        top = tk.Toplevel(root)
        top.title("Aperçu (clic ou Échap pour fermer)")
        img = p["thumb"]
        h = min(root.winfo_screenheight() - 160, 950)
        w = int(img.width * h / img.height)
        tkimg = ImageTk.PhotoImage(img.resize((w, h), Image.LANCZOS))
        lbl = tk.Label(top, image=tkimg)
        lbl.image = tkimg
        lbl.pack()
        lbl.bind("<Button-1>", lambda e: top.destroy())
        top.bind("<Escape>", lambda e: top.destroy())

    def move(i, d):
        pages = st["pages"]
        j = i + d
        if 0 <= j < len(pages):
            pages[i], pages[j] = pages[j], pages[i]
            render()

    def render():
        for w in inner.winfo_children():
            w.destroy()
        if not st["pages"]:
            ttk.Label(inner, text="Choisissez d'abord un PDF : les pages apparaîtront ici.").pack(padx=20, pady=40)
            return
        for i, p in enumerate(st["pages"]):
            card = ttk.Frame(inner, padding=6, relief="solid", borderwidth=1)
            card.grid(row=0, column=i, padx=5, pady=4, sticky="n")
            p["tk"] = ImageTk.PhotoImage(p["small"])
            img = tk.Label(card, image=p["tk"], cursor="hand2")
            img.pack()
            img.bind("<Double-Button-1>", lambda e, p=p: zoom(p))
            ttk.Label(card, text=f"Page {i + 1}", font=("Helvetica", 13, "bold")).pack(pady=(4, 0))
            ttk.Label(card, text=f"n° imprimé : {p['num']}" if p.get("num") else "n° imprimé : non lu",
                      font=("Helvetica", 10)).pack()
            if p.get("mark_range"):
                lo, hi = p["mark_range"]
                a, b = mark_label(p["mark_kind"], lo), mark_label(p["mark_kind"], hi)
                ttk.Label(card, text=f"repères {a}" if lo == hi else f"repères {a} à {b}",
                          font=("Helvetica", 10)).pack()
            else:
                ttk.Label(card, text="repères : aucun", font=("Helvetica", 10)).pack()
            row = ttk.Frame(card)
            row.pack(pady=(2, 0))
            ttk.Button(row, text="◀", width=3, command=lambda i=i: move(i, -1),
                       state="normal" if i > 0 else "disabled").pack(side="left", padx=2)
            ttk.Button(row, text="▶", width=3, command=lambda i=i: move(i, 1),
                       state="normal" if i < len(st["pages"]) - 1 else "disabled").pack(side="left", padx=2)
            ttk.Checkbutton(card, text="Garder", variable=p["keep"]).pack(pady=(4, 0))

    def reapply_auto():
        if not st["pages"]:
            return
        byn = {p["n"]: p for p in st["pages"]}
        st["pages"] = [byn[n] for n in st["auto"] if n in byn]
        render()
        status.set("Pages reclassées d'après les numéros imprimés.")

    # ---------- analyse
    def offer_install():
        if getattr(sys, "frozen", False) or sys.platform != "darwin" or st["asked_install"]:
            return
        st["asked_install"] = True
        if not messagebox.askyesno(
                "Classement automatique",
                "Pour classer les pages d'après leurs numéros, l'application doit installer un petit "
                "composant (environ 10 Mo).\n\nL'installer maintenant ?"):
            return
        set_busy(True, "Installation du composant… (environ une minute)")

        def work():
            r = subprocess.run([sys.executable, "-m", "pip", "install", "pyobjc-framework-Vision"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-700:])
            import importlib
            importlib.invalidate_caches()
            return True

        def done(res, err):
            set_busy(False)
            if err:
                status.set("L'installation a échoué : classement manuel avec les flèches.")
                messagebox.showerror("Installation impossible", str(err))
                return
            get_ocr(retry=True)
            analyze()
        bg(work, done)

    def analyze():
        path = st["path"]
        split = not single.get()
        set_busy(True, "Analyse de la partition en cours… (lecture des numéros de page)")

        def work():
            src = fitz.open(path)
            halves = analyze_scan(src, 0.002, False, light=True, split=split)
            ok = [h for h in halves if not h["blank"]]
            has_ocr = detect_page_numbers(src, ok)
            order = auto_order(ok) if has_ocr else [h["n"] for h in ok]
            return halves, has_ocr, order

        def done(res, err):
            set_busy(False)
            if err:
                status.set("Erreur pendant l'analyse.")
                messagebox.showerror("Erreur", str(err))
                return
            halves, has_ocr, order = res
            pages, blanks = {}, 0
            for h in halves:
                if h["blank"]:
                    blanks += 1
                    continue
                im = h["thumb"]
                small = im.resize((118, int(im.height * 118 / im.width)), Image.LANCZOS)
                pages[h["n"]] = {"n": h["n"], "thumb": im, "small": small, "num": h.get("num"),
                                 "mark_range": h.get("mark_range"), "mark_kind": h.get("mark_kind"),
                                 "keep": tk.BooleanVar(value=True)}
            st["auto"] = list(order)
            st["pages"] = [pages[n] for n in order if n in pages]
            st["analysed"] = path
            render()
            read = sum(1 for p in pages.values() if p["num"])
            marked = sum(1 for p in pages.values() if p["mark_range"])
            msg = f"{len(pages)} page(s) trouvée(s)"
            if blanks:
                msg += f", {blanks} blanche(s) retirée(s)"
            if has_ocr and (read or marked):
                msg += (f". Numéros imprimés lus sur {read} page(s), repères sur {marked} : classement "
                        "automatique fait. Vérifiez les pages sans indication.")
            elif has_ocr:
                msg += ". Ni numéros ni repères lus : classez les pages avec les flèches."
            else:
                msg += ". Lecture des numéros indisponible : classez les pages avec les flèches."
            status.set(msg)
            if not has_ocr:
                offer_install()
        bg(work, done)

    def choose():
        f = filedialog.askopenfilename(title="Choisir la partition", filetypes=[("PDF", "*.pdf")])
        if not f:
            return
        st.update(path=f, pages=[], analysed=None, auto=[])
        file_lbl.config(text=os.path.basename(f))
        render()
        analyze()

    def choose_again():
        if st["path"]:
            st.update(pages=[], analysed=None, auto=[])
            render()
            analyze()

    def open_file(path):
        if sys.platform == "darwin":
            subprocess.run(["open", path])
        elif sys.platform.startswith("win"):
            os.startfile(path)  # noqa
        else:
            subprocess.run(["xdg-open", path])

    def create():
        if not st["path"]:
            messagebox.showinfo("Partition", "Choisissez d'abord un PDF.")
            return
        if st["analysed"] != st["path"]:
            messagebox.showinfo("Patience", "L'analyse de la partition n'est pas terminée.")
            return
        order = [p["n"] for p in st["pages"] if p["keep"].get()]
        if not order:
            messagebox.showwarning("Aucune page", "Cochez au moins une page à garder.")
            return
        default = os.path.splitext(st["path"])[0] + "_A4.pdf"
        out = filedialog.asksaveasfilename(title="Enregistrer la partition A4",
                                           initialdir=os.path.dirname(default),
                                           initialfile=os.path.basename(default),
                                           defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not out:
            return
        set_busy(True, "Création de la partition A4… (quelques secondes par page)")
        path, split = st["path"], not single.get()

        def work():
            return convert(path, out, "scan", order=order, split=split)

        def done(n, err):
            set_busy(False)
            if err:
                status.set("Erreur pendant la création.")
                messagebox.showerror("Erreur", str(err))
                return
            status.set(f"Terminé : {n} page(s) A4 enregistrée(s).")
            if messagebox.askyesno("Terminé", f"Votre partition A4 ({n} pages) est prête.\n\nL'ouvrir maintenant ?"):
                open_file(out)
        bg(work, done)

    render()
    root.mainloop()


def selftest():
    """Auto-test utilisé à la compilation : fabrique une mini-partition (lettres encadrées),
    la fait passer par toute la chaîne et vérifie le résultat."""
    lines, code = [], 0

    def log(msg):
        lines.append(msg)
        print(msg)
    try:
        tmp = tempfile.mkdtemp()
        pdf = os.path.join(tmp, "essai.pdf")
        letters = [["A", "B"], ["C", "D"]]
        d = fitz.open()
        for pg in (1, 0):  # scan mélangé : la page C-D est scannée avant la page A-B
            page = d.new_page(width=841.9, height=1190.5)
            page.insert_text((60, 60), "CLARINET I B", fontsize=10)
            y = 140
            for k in range(2):
                for ln in range(5):
                    page.draw_line((60, y + ln * 8), (780, y + ln * 8), width=0.6)
                for x in range(120, 780, 120):
                    page.draw_line((x, y), (x, y + 32), width=0.8)
                page.draw_rect(fitz.Rect(70, y - 30, 92, y - 8), width=1)
                page.insert_text((75, y - 13), letters[pg][k], fontsize=15, fontname="hebo")
                for x in range(140, 760, 60):
                    page.draw_circle((x, y + 16), 4, fill=(0, 0, 0))
                y += 240
        d.save(pdf)
        src = fitz.open(pdf)
        ok = [h for h in analyze_scan(src, 0.002, False, light=True) if not h["blank"]]
        has_ocr = detect_page_numbers(src, ok)
        log("Lecture des numéros et repères : " + (get_ocr()[0] or "INDISPONIBLE"))
        if has_ocr:
            order = auto_order(ok)
            log(f"Ordre trouvé : {order} (attendu : [2, 1])")
            if order != [2, 1]:
                code = 2
        else:
            code = 2
        n = convert(pdf, os.path.join(tmp, "sortie.pdf"), "scan", order=[2, 1])
        log(f"Conversion : {n} page(s) A4 (attendu : 2)")
        if n != 2:
            code = 1
    except Exception as e:  # noqa
        log(f"ERREUR : {e!r}")
        code = 1
    log("RESULTAT : " + ("OK" if code == 0 else f"A VERIFIER (code {code})"))
    try:
        with open("selftest.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError:
        pass
    return code


if __name__ == "__main__":
    sys.argv = [x for x in sys.argv if not x.startswith("-psn")]  # ancien lancement Finder (macOS)
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if len(sys.argv) == 1:
        gui()
        sys.exit()
    ap = argparse.ArgumentParser(description="Partition PDF A3 -> A4 optimisé")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--mode", choices=["reflow", "split", "fit", "scan"], default="reflow")
    ap.add_argument("--margin", type=float, default=8.0, help="marges A4 en mm")
    ap.add_argument("--spacing", type=float, default=6.0, help="espace entre systèmes (mm, reflow)")
    ap.add_argument("--gap", type=float, default=12.0,
                    help="blanc minimum entre deux systèmes (mm, reflow). Augmenter si un système "
                         "à deux portées est coupé, diminuer si deux systèmes sont collés.")
    ap.add_argument("--no-justify", action="store_true", help="ne pas étirer l'espacement pour remplir les pages")
    ap.add_argument("--order", default=None,
                    help="(scan) 'auto' = classement d'après les numéros de page imprimés, ou ordre "
                         "manuel des demi-feuilles, ex. 2,6,7,8,5,4 (numéros affichés par le script).")
    ap.add_argument("--single", action="store_true",
                    help="(scan) une seule page par feuille : ne pas couper les feuilles paysage en deux")
    ap.add_argument("--preview", default=None, help="(scan) écrit une planche PNG numérotée des demi-feuilles")
    ap.add_argument("--blank-threshold", type=float, default=0.002, help="(scan) part d'encre sous laquelle une page est blanche")
    ap.add_argument("--no-deskew", action="store_true", help="(scan) ne pas redresser")
    ap.add_argument("--no-clean", action="store_true", help="(scan) ne pas nettoyer le fond")
    ap.add_argument("--dpi", type=int, default=300, help="(scan) résolution de sortie")
    a = ap.parse_args()
    order = "auto" if a.order == "auto" else ([int(x) for x in a.order.split(",")] if a.order else None)
    n = convert(a.input, a.output, a.mode, a.margin, a.spacing, a.gap, not a.no_justify,
                order, a.blank_threshold, not a.no_deskew, not a.no_clean, a.dpi, a.preview, not a.single)
    print(f"{n} page(s) A4 -> {a.output}")
