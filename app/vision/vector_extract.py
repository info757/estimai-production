"""Vector-first token extraction from PDF pages using PyMuPDF (fitz).

Captures exact tokens for sanitary runs on profile sheets, including:
 - length_text (e.g., "117 LF") and parsed length_ft
 - diameter (e.g., 8") and material (PVC/DIP/etc.)
 - optional slope text (e.g., "@ 0.50%")

This module avoids rasterization for numeric values and returns
deterministic results suitable for aggregation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF


DIAMETER_RE = re.compile(r"(?P<dia>\d{1,2})\s*\"", re.I)
LENGTH_RE = re.compile(r"(?P<len>\d+(?:\.\d+)?)\s*LF\b", re.I)
SLOPE_RE = re.compile(r"@\s*(?P<slope>\d+(?:\.\d+)?)%", re.I)
# Enhanced material regex to catch D.I.P., DUCTILE IRON, etc.
MATERIAL_RE = re.compile(r"\b(PVC|DIP|D\.I\.P\.?|DUCTILE\s*IRON|RCP|HDPE)\b", re.I)


def _normalize_material(token: str) -> str:
    """Normalize material tokens, handling OCR slips and abbreviations."""
    if not token:
        return None
    t = token.upper().replace('.', '').replace(' ', '').replace('-', '')
    # DIP variations
    if t in {"DIP", "DIP", "DUCTILEIRON", "DUCTILEIRONPIPE", "D1P", "D|P", "DIPPIPE", "SIP", "DI"}:
        return "DIP"
    # Other materials
    if t in {"PVC"}:
        return "PVC"
    if t in {"RCP", "RCPP"}:
        return "RCP"
    if t in {"HDPE"}:
        return "HDPE"
    return token.upper()


@dataclass
class VectorRun:
    raw: str
    length_text: Optional[str]
    length_ft: Optional[float]
    diameter_text: Optional[str]
    material: Optional[str]
    slope_text: Optional[str]
    bbox: tuple


def _page_text_spans(doc: fitz.Document, page_index: int) -> List[Dict[str, Any]]:
    page = doc.load_page(page_index)
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type, ...) per block
    spans: List[Dict[str, Any]] = []
    for b in blocks:
        x0, y0, x1, y1, text = b[:5]
        if not text or not text.strip():
            continue
        spans.append({
            "bbox": (x0, y0, x1, y1),
            "text": text.strip()
        })
    return spans


def extract_profile_runs_from_text(pdf_path: str, page_number_1_indexed: int) -> List[VectorRun]:
    """Extract sanitary profile run tokens from vector text on a given page.

    Args:
        pdf_path: absolute path to PDF
        page_number_1_indexed: 1-based page number

    Returns:
        List of VectorRun with exact tokens
    """
    runs: List[VectorRun] = []
    with fitz.open(pdf_path) as doc:
        page_idx = page_number_1_indexed - 1
        spans = _page_text_spans(doc, page_idx)

    # Heuristic: lines that contain both a length token and a diameter/material token
    for s in spans:
        text = " ".join(s["text"].split())
        m_len = LENGTH_RE.search(text)
        m_dia = DIAMETER_RE.search(text)
        m_mat = MATERIAL_RE.search(text)
        m_slope = SLOPE_RE.search(text)

        if m_len and (m_dia or m_mat):
            length_text = m_len.group(0)
            try:
                length_ft = float(m_len.group("len"))
            except Exception:
                length_ft = None
            diameter_text = m_dia.group(0) if m_dia else None
            # Normalize material to handle D.I.P., DUCTILE IRON, etc.
            material_raw = m_mat.group(1) if m_mat else None
            material = _normalize_material(material_raw) if material_raw else None
            slope_text = m_slope.group(0) if m_slope else None
            
            # Also check for DIP patterns if regex didn't match (e.g., "DUCTILE IRON" in text)
            if not material and "DUCTILE" in text.upper() and "IRON" in text.upper():
                material = "DIP"

            runs.append(VectorRun(
                raw=text,
                length_text=length_text,
                length_ft=length_ft,
                diameter_text=diameter_text,
                material=material,
                slope_text=slope_text,
                bbox=s["bbox"],
            ))

    return runs


