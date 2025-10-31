"""
Test script for processing a single sheet (e.g., C-2.1) from Dawn Ridge PDF.

Finds the page containing the specified sheet number, processes only that page,
and generates accuracy results.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vision import UniversalVisionAgent, parse_markdown_to_json
from app.evaluation.custom_metrics import evaluate_takeoff_custom
import fitz  # PyMuPDF for PDF text extraction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_sheet_page(pdf_path: str, sheet_number: str) -> Optional[int]:
    """
    Find the page number (0-indexed) containing the specified sheet number.
    
    Args:
        pdf_path: Path to PDF file
        sheet_number: Sheet number to find (e.g., "C-2.1")
        
    Returns:
        Page number (0-indexed) if found, None otherwise
    """
    logger.info(f"Searching for sheet '{sheet_number}' in {pdf_path}")
    
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().upper()
        
        # Look for sheet number in text (handle variations like "C-2.1", "C 2.1", etc.)
        search_patterns = [
            sheet_number.upper(),
            sheet_number.replace("-", " ").upper(),  # "C 2.1"
            sheet_number.replace("-", "").upper(),   # "C2.1"
            f"SHEET {sheet_number.upper()}",
            f"SHT {sheet_number.upper()}",
        ]
        
        for pattern in search_patterns:
            if pattern in text:
                logger.info(f"Found sheet '{sheet_number}' on page {page_num + 1} (0-indexed: {page_num})")
                doc.close()
                return page_num
    
    doc.close()
    logger.warning(f"Sheet '{sheet_number}' not found in PDF")
    return None


async def test_single_sheet(
    pdf_path: str,
    sheet_number: str,
    firm: str = "hagen_engineering",
    use_three_pass: bool = True
) -> Dict[str, Any]:
    """
    Test extraction on a single sheet/page.
    
    Args:
        pdf_path: Path to PDF file
        sheet_number: Sheet number to process (e.g., "C-2.1")
        firm: Firm identifier
        use_three_pass: Use three-pass workflow
        
    Returns:
        Dictionary with test results
    """
    logger.info("="*80)
    logger.info(f"TESTING SINGLE SHEET: {sheet_number}")
    logger.info("="*80)
    
    # Find the page with this sheet number
    page_index = find_sheet_page(pdf_path, sheet_number)
    if page_index is None:
        raise ValueError(f"Sheet '{sheet_number}' not found in PDF")
    
    page_number = page_index + 1  # Convert to 1-indexed for display
    logger.info(f"Processing page {page_number} (0-indexed: {page_index})")
    
    # Initialize Universal Vision Agent
    logger.info("\nInitializing Universal Vision Agent...")
    agent = UniversalVisionAgent()
    
    # Process only this page
    logger.info(f"\nAnalyzing page {page_number} with sheet {sheet_number}...")
    
    start_time = datetime.now()
    
    # Use page_range to process only this page (1-indexed for the API)
    results = await agent.analyze_document(
        pdf_path=pdf_path,
        firm=firm,
        auto_detect_firm=True,
        use_three_pass=use_three_pass,
        page_range=[page_number]  # 1-indexed
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"\nAnalysis completed in {duration:.1f} seconds")
    logger.info(f"  - Firm detected: {results['firm_detected']}")
    logger.info(f"  - Pages processed: {results['metadata']['total_pages']}")
    
    # Parse markdown to JSON
    logger.info("\nParsing markdown to structured JSON...")
    predicted_data = parse_markdown_to_json(results["markdown"])
    
    # Print extraction summary
    logger.info(f"\nExtraction Summary for Sheet {sheet_number}:")
    logger.info(f"  - Pipes: {len(predicted_data.get('pipes', []))}")
    logger.info(f"  - Structures: {len(predicted_data.get('structures', []))}")
    logger.info(f"  - Earthwork: {len(predicted_data.get('earthwork', []))}")
    
    # Calculate totals
    total_lf = sum(p.get('length_ft', 0) * p.get('count', 1) for p in predicted_data.get('pipes', []))
    logger.info(f"  - Total LF: {total_lf:.2f}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Save markdown
    markdown_path = output_dir / f"sheet_{sheet_number.replace('-', '_')}_extraction.md"
    with open(markdown_path, 'w') as f:
        f.write(results["markdown"])
    logger.info(f"\nSaved markdown extraction to: {markdown_path}")
    
    # Save JSON
    json_path = output_dir / f"sheet_{sheet_number.replace('-', '_')}_extraction.json"
    with open(json_path, 'w') as f:
        json.dump(predicted_data, f, indent=2)
    logger.info(f"Saved parsed JSON to: {json_path}")
    
    # Generate summary report
    report = f"""# Single Sheet Test Report: {sheet_number}

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Sheet Number**: {sheet_number}
**Page Number**: {page_number}
**Firm**: {results['firm_detected']}
**Processing Time**: {duration:.1f} seconds

---

## Extraction Results

### Pipes
- **Count**: {len(predicted_data.get('pipes', []))}
- **Total Linear Feet**: {total_lf:.2f} LF

### Structures
- **Count**: {len(predicted_data.get('structures', []))}

### Earthwork
- **Count**: {len(predicted_data.get('earthwork', []))}

---

## Extracted Pipes

"""
    
    for i, pipe in enumerate(predicted_data.get('pipes', [])[:10], 1):
        report += f"{i}. {pipe.get('diameter_in', 'N/A')}\" {pipe.get('material', 'N/A')} - {pipe.get('length_ft', 0):.2f} LF\n"
    
    if len(predicted_data.get('pipes', [])) > 10:
        report += f"\n... and {len(predicted_data.get('pipes', [])) - 10} more pipes\n"
    
    report_path = output_dir / f"sheet_{sheet_number.replace('-', '_')}_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Saved report to: {report_path}")
    
    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)
    
    return {
        "sheet_number": sheet_number,
        "page_number": page_number,
        "results": predicted_data,
        "metadata": results["metadata"],
        "duration": duration,
        "output_files": {
            "markdown": str(markdown_path),
            "json": str(json_path),
            "report": str(report_path)
        }
    }


# =============================================================================
# Main Execution
# =============================================================================

async def main():
    """Main entry point."""
    # Default to C-2.1 if not specified
    sheet_number = sys.argv[1] if len(sys.argv) > 1 else "C-2.1"
    
    base_dir = Path(__file__).parent.parent
    pdf_path = base_dir / "data/test_pdfs/Dawn Ridge Homes_HEPA_Combined_04-1-25.pdf"
    
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    try:
        results = await test_single_sheet(
            pdf_path=str(pdf_path),
            sheet_number=sheet_number,
            firm="hagen_engineering",
            use_three_pass=True
        )
        
        logger.info(f"\n✅ Successfully processed sheet {sheet_number}")
        logger.info(f"📄 Results saved to: {results['output_files']['report']}")
        
    except Exception as e:
        logger.error(f"Error testing sheet {sheet_number}: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
