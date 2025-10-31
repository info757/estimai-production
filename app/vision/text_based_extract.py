"""
Unified text-based extraction module for sewer pipes.

Combines vector text extraction (PyMuPDF) and OCR (pytesseract) to extract
pipe callouts from construction drawings without using vision robotics.

Strategy:
1. Try vector extraction first (fastest, most accurate)
2. Fall back to OCR if vector fails or finds too few runs
3. Merge and deduplicate results intelligently
4. Aggregate by material and diameter
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.vision.vector_extract import extract_profile_runs_from_text, VectorRun
from app.vision.ocr_extract import ocr_profile_runs_strict_segments

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPipe:
    """Aggregated pipe information."""
    diameter_in: float
    material: str
    total_length_ft: float
    count: int
    source: str  # "vector", "ocr", or "merged"


def extract_sewer_pipes(
    pdf_path: str,
    page_num: int,
    min_runs_threshold: int = 2,
    dpi: int = 450
) -> Dict[str, Any]:
    """
    Extract sewer pipes from a PDF page using vector text and OCR.
    
    Args:
        pdf_path: Path to PDF file
        page_num: 1-indexed page number
        min_runs_threshold: Minimum runs to accept from vector extraction before trying OCR
        dpi: DPI for OCR (higher = more accurate but slower)
        
    Returns:
        Dictionary with:
        - "pipes": List of ExtractedPipe objects
        - "vector_runs": List of vector-extracted runs
        - "ocr_runs": List of OCR-extracted runs (if used)
        - "source": "vector", "ocr", or "merged"
    """
    logger.info(f"Extracting sewer pipes from page {page_num}")
    
    # Step 1: Try vector extraction first (fastest, most accurate)
    logger.info("Attempting vector text extraction...")
    vector_runs = extract_profile_runs_from_text(pdf_path, page_num, debug=True)
    
    logger.info(f"Vector extraction found {len(vector_runs)} runs")
    
    # Step 2: If vector fails or finds too few, try OCR
    ocr_runs: List[Dict[str, Any]] = []
    dpi_candidates = [dpi]
    # Add fallbacks commonly successful for these drawings
    for fallback_dpi in (350, 300):
        if fallback_dpi not in dpi_candidates:
            dpi_candidates.append(fallback_dpi)

    if len(vector_runs) < min_runs_threshold:
        logger.info(f"Vector found < {min_runs_threshold} runs, trying OCR (DPIs: {dpi_candidates})...")
        for current_dpi in dpi_candidates:
            try:
                current_runs = ocr_profile_runs_strict_segments(pdf_path, page_num, dpi=current_dpi)
                if current_runs:
                    logger.info("OCR extraction at %s DPI found %s runs", current_dpi, len(current_runs))
                    ocr_runs.extend(current_runs)
                else:
                    logger.info("OCR extraction at %s DPI returned no runs", current_dpi)
            except Exception as e:
                logger.warning("OCR extraction failed at %s DPI: %s", current_dpi, e)
    else:
        logger.info("Vector runs sufficient (%s >= %s); skipping OCR fallback", len(vector_runs), min_runs_threshold)

    # Deduplicate OCR runs collected from multiple DPI passes
    if ocr_runs:
        unique_runs: List[Dict[str, Any]] = []
        for run in ocr_runs:
            length_ft = run.get("length_ft")
            material = run.get("material")
            if length_ft is None:
                continue
            duplicate = False
            for existing in unique_runs:
                existing_length = existing.get("length_ft")
                if existing_length is None:
                    continue
                if abs(existing_length - length_ft) < 1:
                    existing_material = existing.get("material")
                    if existing_material is None and material:
                        existing["material"] = material
                    if existing_material == material or existing_material is None or material is None:
                        duplicate = True
                        break
            if not duplicate:
                unique_runs.append(run)
        ocr_runs = unique_runs
        logger.info("OCR deduped to %s unique runs", len(ocr_runs))
    
    # Step 3: Merge and deduplicate results
    if vector_runs and ocr_runs:
        logger.info("Merging vector and OCR results...")
        merged_runs = _merge_runs(vector_runs, ocr_runs)
        source = "merged"
    elif vector_runs:
        merged_runs = vector_runs
        source = "vector"
    elif ocr_runs:
        merged_runs = ocr_runs
        source = "ocr"
    else:
        logger.warning("No runs found from either vector or OCR extraction")
        return {
            "pipes": [],
            "vector_runs": [],
            "ocr_runs": [],
            "source": "none"
        }
    
    # Step 4: Aggregate by material and diameter
    logger.info("Aggregating pipes by diameter and material...")
    aggregated_pipes = _aggregate_pipes(merged_runs, source)
    
    logger.info(f"Extracted {len(aggregated_pipes)} unique pipe types")
    
    return {
        "pipes": aggregated_pipes,
        "vector_runs": [
            {
                "length_ft": r.length_ft,
                "diameter_text": r.diameter_text,
                "material": r.material,
                "raw": r.raw
            }
            for r in vector_runs
        ],
        "ocr_runs": [
            {
                "length_ft": r.get("length_ft"),
                "diameter_text": r.get("diameter_text"),
                "material": r.get("material"),
                "raw": r.get("raw")
            }
            for r in ocr_runs
        ],
        "source": source
    }


def _merge_runs(vector_runs: List[VectorRun], ocr_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge vector and OCR runs, deduplicating similar entries.
    
    Prefers vector over OCR when duplicates are found (vector is more accurate).
    
    Args:
        vector_runs: Vector-extracted runs
        ocr_runs: OCR-extracted runs
        
    Returns:
        List of merged runs (dict format)
    """
    merged = []
    
    # Add all vector runs first
    for run in vector_runs:
        merged.append({
            "length_ft": run.length_ft,
            "diameter_text": run.diameter_text,
            "material": run.material,
            "raw": run.raw,
            "source": "vector"
        })
    
    # Add OCR runs, checking for duplicates
    for ocr_run in ocr_runs:
        ocr_length = ocr_run.get("length_ft")
        ocr_material = ocr_run.get("material")
        ocr_dia = ocr_run.get("diameter_text")
        
        if not ocr_length:
            continue
        
        # Check if this OCR run is similar to an existing vector run
        is_duplicate = False
        for merged_run in merged:
            merged_length = merged_run.get("length_ft")
            merged_material = merged_run.get("material")

            if merged_length is None:
                continue

            length_diff = abs(ocr_length - merged_length)
            materials_match = (
                ocr_material == merged_material or
                (ocr_material is None and merged_material is None)
            )

            if length_diff < 2 and (materials_match or ocr_material is None or merged_material is None):
                # Promote material information if one run has it and the other doesn't
                if merged_material is None and ocr_material:
                    merged_run["material"] = ocr_material
                is_duplicate = True
                break
        
        if not is_duplicate:
            merged.append({
                "length_ft": ocr_length,
                "diameter_text": ocr_dia,
                "material": ocr_material,
                "raw": ocr_run.get("raw"),
                "source": "ocr"
            })
    
    logger.info(f"Merged {len(vector_runs)} vector + {len(ocr_runs)} OCR = {len(merged)} unique runs")
    return merged


def _aggregate_pipes(runs: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """
    Aggregate pipe runs by diameter and material.
    
    Args:
        runs: List of pipe run dictionaries
        source: Source of the runs ("vector", "ocr", or "merged")
        
    Returns:
        List of aggregated pipe dictionaries
    """
    # Group by (diameter, material)
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    
    for run in runs:
        length_ft = run.get("length_ft")
        if not length_ft:
            continue
        
        # Extract diameter from diameter_text (e.g., "8\"" -> 8)
        diameter_text = run.get("diameter_text", "")
        raw_text = run.get("raw", "") or ""
        import re
        diameter = None
        if raw_text:
            near_lf = re.search(r'LF[^\d]{0,6}(\d{1,2})\s*(?:"|”|″)', raw_text, re.I)
            if near_lf:
                diameter = float(near_lf.group(1))
        if diameter is None and diameter_text:
            import re
            try:
                dia_match = re.search(r'(\d{1,2})', diameter_text)
                if dia_match:
                    diameter = float(dia_match.group(1))
            except Exception:
                diameter = None
        if diameter is None and raw_text:
            dia_match = re.search(r'(\d{1,2})\s*(?:"|”|″)', raw_text)
            if dia_match:
                diameter = float(dia_match.group(1))

        # Filter out obviously invalid diameters (e.g., OCR artifacts)
        if diameter is not None and diameter > 60:
            logger.debug("Skipping unrealistic diameter %.1f from raw '%s'", diameter, raw_text)
            diameter = None
        
        material = run.get("material", "Unknown")
        
        key = (diameter, material)
        if key not in groups:
            groups[key] = []
        groups[key].append(run)
    
    # Aggregate each group
    aggregated = []
    for (diameter, material), group_runs in groups.items():
        total_length = sum(r.get("length_ft", 0) for r in group_runs)
        count = len(group_runs)
        
        aggregated.append({
            "diameter_in": diameter,
            "material": material or "Unknown",
            "length_ft": total_length,
            "count": count,
            "source": source
        })
    
    return aggregated

