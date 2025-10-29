"""
Fast Test Script - Test on First 3 Pages Only

Tests the system on just the first 3 pages to verify functionality
without the 45-minute wait time.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vision import UniversalVisionAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_fast_test():
    """
    Run fast test on first 3 pages of Dawn Ridge PDF.
    
    This should take 2-3 minutes instead of 45 minutes.
    """
    logger.info("="*80)
    logger.info("FAST TEST: Dawn Ridge Homes (First 3 Pages)")
    logger.info("="*80)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    pdf_path = base_dir / "data/test_pdfs/Dawn Ridge Homes_HEPA_Combined_04-1-25.pdf"
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize Universal Vision Agent
    logger.info("\nInitializing Universal Vision Agent...")
    agent = UniversalVisionAgent()
    
    # Run analysis on first 3 pages only
    logger.info(f"\nAnalyzing PDF: {pdf_path.name} (Page 1 only)")
    logger.info("This should take 1-2 minutes...")
    
    start_time = datetime.now()
    
    results = await agent.analyze_document(
        pdf_path=str(pdf_path),
        firm="hagen_engineering",
        auto_detect_firm=True,
        use_three_pass=True,  # Three-pass for accuracy
        page_range=[1]  # Only first page for initial test
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"\nAnalysis completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"  - Firm detected: {results['firm_detected']}")
    logger.info(f"  - Pages processed: {results['metadata']['total_pages']}")
    
    # Save natural language output
    output_path = output_dir / "page1_extraction.txt"
    with open(output_path, 'w') as f:
        f.write(results["markdown"])
    logger.info(f"\nSaved extraction to: {output_path}")
    logger.info("\nManually review the output to verify it captured all construction data.")
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("BASELINE TEST COMPLETE")
    logger.info("="*80)
    logger.info(f"Processing Time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Output saved to: {output_path}")
    logger.info("\nNext steps:")
    logger.info("1. Review the output file to verify extraction quality")
    logger.info("2. Count pipes and structures manually")
    logger.info("3. Compare to ground truth")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_fast_test())
