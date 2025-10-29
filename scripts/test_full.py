#!/usr/bin/env python3
"""Run full 25-page extraction test."""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

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


async def run_full_test():
    """Run extraction on all 25 pages."""
    logger.info("="*80)
    logger.info("FULL TEST: Dawn Ridge Homes (All 25 Pages)")
    logger.info("="*80)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    pdf_path = base_dir / "data/test_pdfs/Dawn Ridge Homes_HEPA_Combined_04-1-25.pdf"
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize Universal Vision Agent
    logger.info("\nInitializing Universal Vision Agent...")
    agent = UniversalVisionAgent()
    
    # Run analysis on all pages
    logger.info(f"\nAnalyzing PDF: {pdf_path.name} (All 25 pages)")
    logger.info("This will take approximately 45 minutes...")
    
    start_time = datetime.now()
    
    results = await agent.analyze_document(
        pdf_path=str(pdf_path),
        firm="hagen_engineering",
        auto_detect_firm=True,
        use_three_pass=True,  # Three-pass for accuracy
        page_range=None  # All pages
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"\nAnalysis completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"  - Firm detected: {results['firm_detected']}")
    logger.info(f"  - Pages processed: {results['metadata']['total_pages']}")
    
    # Save natural language output
    output_path = output_dir / "full_extraction.txt"
    with open(output_path, 'w') as f:
        f.write(results["markdown"])
    logger.info(f"\nSaved extraction to: {output_path}")
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("FULL TEST COMPLETE")
    logger.info("="*80)
    logger.info(f"Processing Time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"Output saved to: {output_path}")
    logger.info("\nNext steps:")
    logger.info("1. Compare extraction to ground truth (29 pipes expected)")
    logger.info("2. Use LLM to score accuracy")
    logger.info("3. Document baseline results")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_full_test())

