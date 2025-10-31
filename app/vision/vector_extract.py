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
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

DIAMETER_RE = re.compile(r"(?P<dia>\d{1,2})\s*(\"|”|″|“)", re.I)
LENGTH_RE = re.compile(r"(?P<len>\d+(?:\.\d+)?)\s*LF\b", re.I)
SLOPE_RE = re.compile(r"@\s*(?P<slope>\d+(?:\.\d+)?)%", re.I)
# Enhanced material regex to catch D.I.P., DUCTILE IRON, etc., and OCR errors
MATERIAL_RE = re.compile(r"\b(PVC|DIP|D\.I\.P\.?|D\.1\.P\.?|DUCTILE\s*IRON|RCP|HDPE|PNY)\b", re.I)


def _normalize_material(token: str) -> str:
    """Normalize material tokens, handling OCR slips and abbreviations."""
    if not token:
        return None
    t = token.upper().replace('.', '').replace(' ', '').replace('-', '')
    # DIP variations (including common OCR errors)
    dip_variants = {
        "DIP", "DUCTILEIRON", "DUCTILEIRONPIPE", "D1P", "D|P", "DIPPIPE", 
        "SIP", "DI", "DIP", "DIPP"
    }
    if t in dip_variants:
        return "DIP"
    # PVC variations (including OCR errors)
    if t in {"PVC", "PNY", "PVC,", "PVC=", "PVG", "PYC"}:  # OCR slips
        return "PVC"
    # Other materials
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


def extract_profile_runs_from_text(
    pdf_path: str,
    page_number_1_indexed: int,
    debug: bool = False,
) -> List[VectorRun]:
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

    if debug:
        logger.info("Vector extraction: %s spans found on page %s", len(spans), page_number_1_indexed)

    # Heuristic: lines that contain both a length token and a diameter/material token
    for s in spans:
        text = " ".join(s["text"].split())
        m_len = LENGTH_RE.search(text)
        m_dia = DIAMETER_RE.search(text)
        m_mat = MATERIAL_RE.search(text)
        m_slope = SLOPE_RE.search(text)

        # Must have length
        if not m_len:
            if debug:
                logger.debug("Skipping span without length token: %s", text[:80])
            continue
        
        # Must have diameter OR material
        if not (m_dia or m_mat):
            # Try harder to find material - check for DIP patterns
            text_upper = text.upper()
            has_dip_indicator = any([
                "DUCTILE" in text_upper and "IRON" in text_upper,
                "D.I.P" in text_upper or "D.1.P" in text_upper,
                "SIP" in text_upper,  # OCR error for DIP
            ])
            if not has_dip_indicator:
                if debug:
                    logger.debug("Skipping span without diameter/material: %s", text[:80])
                continue
        
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
        if not material:
            text_upper = text.upper()
            if "DUCTILE" in text_upper and "IRON" in text_upper:
                material = "DIP"
            elif "D.I.P" in text_upper or "D.1.P" in text_upper or "SIP" in text_upper:
                material = "DIP"

        runs.append(
            VectorRun(
                raw=text,
                length_text=length_text,
                length_ft=length_ft,
                diameter_text=diameter_text,
                material=material,
                slope_text=slope_text,
                bbox=s["bbox"],
            )
        )

    if debug:
        logger.info("Vector extraction: %s runs detected", len(runs))

    return runs


