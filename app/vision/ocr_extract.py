"""OCR fallback to read profile callouts like '117 LF 8" PVC @ 0.50%'.

Uses pdf2image to render a single page at high DPI and pytesseract to
recover small text when the PDF has no usable vector text layer.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any

from pdf2image import convert_from_path
import pytesseract


LENGTH_RE = re.compile(r"(?P<len>\d+(?:\.\d+)?)\s*LF\b", re.I)
DIAMETER_RE = re.compile(r"(?P<dia>\d{1,2})\s*\"", re.I)
MATERIAL_RE = re.compile(r"\b(PVC|DIP|RCP|HDPE)\b", re.I)
SLOPE_RE = re.compile(r"@\s*(?P<slope>\d+(?:\.\d+)?)%", re.I)


def _normalize_material(token: str) -> str:
    t = (token or '').upper().replace('.', '').replace(' ', '')
    if t in {"DIP", "DUCTILEIRON", "DUCTILEIRONPIPE", "D1P", "D|P", "DIPPIPE", "SIP"}:  # treat SIP as DIP OCR slip
        return "DIP"
    if t in {"PVC"}:
        return "PVC"
    if t in {"RCP"}:
        return "RCP"
    if t in {"HDPE"}:
        return "HDPE"
    return token.upper() if token else token


def ocr_profile_runs(pdf_path: str, page_number_1_indexed: int, dpi: int = 400) -> List[Dict[str, Any]]:
    images = convert_from_path(pdf_path, dpi=dpi, first_page=page_number_1_indexed, last_page=page_number_1_indexed)
    if not images:
        return []
    img = images[0]

    # Use a config that improves small engineering text recognition
    config = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
    # Reconstruct line-wise strings
    lines: Dict[int, str] = {}
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        line_no = data["line_num"][i]
        lines.setdefault(line_no, "")
        if lines[line_no]:
            lines[line_no] += " "
        lines[line_no] += text

    runs: List[Dict[str, Any]] = []
    for line in lines.values():
        norm = " ".join(line.split())
        m_len = LENGTH_RE.search(norm)
        if not m_len:
            continue
        m_dia = DIAMETER_RE.search(norm)
        # Material can be noisy; allow aliases and slips
        m_mat = MATERIAL_RE.search(norm)
        m_slope = SLOPE_RE.search(norm)
        if not (m_dia or m_mat):
            # Likely unrelated LF (e.g., station grid); skip
            continue
        try:
            length_ft = float(m_len.group("len"))
        except Exception:
            length_ft = None
        mat = _normalize_material(m_mat.group(1) if m_mat else None)
        runs.append({
            "raw": norm,
            "length_text": m_len.group(0),
            "length_ft": length_ft,
            "diameter_text": m_dia.group(0) if m_dia else None,
            "material": mat,
            "slope_text": m_slope.group(0) if m_slope else None,
        })

    return runs


def ocr_profile_runs_strict_segments(pdf_path: str, page_number_1_indexed: int, dpi: int = 450) -> List[Dict[str, Any]]:
    """Stricter heuristic: specifically capture 'NNN LF 8" (PVC|DIP) ...' patterns.

    Useful when we want only per-segment callouts and to ignore totals/tables.
    """
    images = convert_from_path(pdf_path, dpi=dpi, first_page=page_number_1_indexed, last_page=page_number_1_indexed)
    if not images:
        return []
    img = images[0]
    config = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(img, config=config)
    lines = [" ".join(l.split()) for l in text.splitlines() if l.strip()]
    runs: List[Dict[str, Any]] = []
    for line in lines:
        norm = line.upper()
        if "LF" not in norm:
            continue
        mlen = LENGTH_RE.search(norm)
        mdia = DIAMETER_RE.search(norm)
        # Accept material tokens or common slips
        mat = None
        if "DIP" in norm or "D.I.P" in norm or "DUCTILE" in norm or "SIP" in norm:
            mat = "DIP"
        elif "PVC" in norm:
            mat = "PVC"
        if not (mlen and (mdia or mat)):
            continue
        try:
            lf = float(mlen.group("len"))
        except Exception:
            lf = None
        dia = mdia.group(0) if mdia else None
        runs.append({
            "raw": line,
            "length_text": mlen.group(0) if mlen else None,
            "length_ft": lf,
            "diameter_text": dia,
            "material": mat,
            "slope_text": (SLOPE_RE.search(norm).group(0) if SLOPE_RE.search(norm) else None),
        })
    return runs


