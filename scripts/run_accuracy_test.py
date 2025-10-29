"""
Accuracy Testing Script for Dawn Ridge (Hagen Engineering)

Tests the Universal Vision Agent against ground truth from Excel spreadsheets.
Evaluates using both RAGAS and custom construction metrics.
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

from app.vision import UniversalVisionAgent, parse_markdown_to_json
from app.evaluation.ragas_eval import RAGASEvaluator
from app.evaluation.custom_metrics import evaluate_takeoff_custom

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_accuracy_test():
    """
    Run accuracy test on Dawn Ridge PDF (Hagen Engineering).
    
    Workflow:
    1. Run Universal Vision Agent on PDF
    2. Parse markdown to JSON
    3. Compare against ground truth
    4. Calculate RAGAS metrics
    5. Calculate custom construction metrics
    6. Generate report
    """
    logger.info("="*80)
    logger.info("ACCURACY TEST: Dawn Ridge Homes (Hagen Engineering)")
    logger.info("="*80)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    pdf_path = base_dir / "data/test_pdfs/Dawn Ridge Homes_HEPA_Combined_04-1-25.pdf"
    ground_truth_path = base_dir / "data/ground_truth/dawn_ridge_annotations.json"
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Load ground truth
    logger.info(f"Loading ground truth from: {ground_truth_path}")
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    logger.info(f"Ground Truth Summary:")
    logger.info(f"  - Pipes: {len(ground_truth.get('expected_pipes', []))}")
    logger.info(f"  - Materials: {len(ground_truth.get('expected_materials', []))}")
    logger.info(f"  - Volume Items: {len(ground_truth.get('expected_volumes', []))}")
    
    # Calculate ground truth totals
    total_lf = sum(p.get('length_ft', 0) * p.get('count', 1) for p in ground_truth.get('expected_pipes', []))
    logger.info(f"  - Total LF: {total_lf:.2f}")
    
    # Initialize Universal Vision Agent
    logger.info("\nInitializing Universal Vision Agent...")
    agent = UniversalVisionAgent()
    
    # Run analysis
    logger.info(f"\nAnalyzing PDF: {pdf_path.name}")
    logger.info("This may take several minutes for a 25-page document...")
    
    start_time = datetime.now()
    
    results = await agent.analyze_document(
        pdf_path=str(pdf_path),
        firm="hagen_engineering",
        auto_detect_firm=True,
        use_three_pass=True
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"\nAnalysis completed in {duration:.1f} seconds")
    logger.info(f"  - Firm detected: {results['firm_detected']}")
    logger.info(f"  - Pages processed: {results['metadata']['total_pages']}")
    
    # Save raw markdown
    markdown_output_path = output_dir / "dawn_ridge_extraction.md"
    with open(markdown_output_path, 'w') as f:
        f.write(results["markdown"])
    logger.info(f"\nSaved markdown extraction to: {markdown_output_path}")
    
    # Parse markdown to JSON
    logger.info("\nParsing markdown to structured JSON...")
    predicted_data = parse_markdown_to_json(results["markdown"])
    
    # Save parsed JSON
    json_output_path = output_dir / "dawn_ridge_extraction.json"
    with open(json_output_path, 'w') as f:
        json.dump(predicted_data, f, indent=2)
    logger.info(f"Saved parsed JSON to: {json_output_path}")
    
    # Print extraction summary
    logger.info(f"\nSystem Extraction Summary:")
    logger.info(f"  - Pipes: {len(predicted_data.get('pipes', []))}")
    logger.info(f"  - Structures: {len(predicted_data.get('structures', []))}")
    logger.info(f"  - Earthwork: {len(predicted_data.get('earthwork', []))}")
    
    # Calculate system totals
    system_lf = sum(p.get('length_ft', 0) * p.get('count', 1) for p in predicted_data.get('pipes', []))
    logger.info(f"  - Total LF: {system_lf:.2f}")
    
    # Evaluate with custom metrics
    logger.info("\n" + "="*80)
    logger.info("CUSTOM CONSTRUCTION METRICS")
    logger.info("="*80)
    
    # Create dummy retrieved contexts for evaluation
    retrieved_contexts = [results["markdown"][:1000]]  # Use first 1000 chars of extraction
    
    custom_results = evaluate_takeoff_custom(predicted_data, ground_truth, retrieved_contexts)
    
    logger.info(f"\nAccuracy Results:")
    logger.info(f"  - Pipe Count Accuracy: {custom_results['pipe_count_accuracy']:.1%}")
    logger.info(f"  - Total LF Accuracy: {custom_results['total_lf_accuracy']:.1%}")
    logger.info(f"  - Material Accuracy: {custom_results['material_accuracy']:.1%}")
    logger.info(f"  - Depth Extraction Rate: {custom_results['depth_extraction_rate']:.1%}")
    logger.info(f"  - Elevation Accuracy: {custom_results.get('elevation_accuracy', 0):.1%}")
    
    # Evaluate with RAGAS
    logger.info("\n" + "="*80)
    logger.info("RAGAS METRICS")
    logger.info("="*80)
    
    try:
        ragas_evaluator = RAGASEvaluator()
        
        # Create RAGAS-compatible format
        ragas_input = {
            "question": "Extract all utility pipes, structures, and earthwork from this construction document",
            "answer": results["markdown"],
            "contexts": [results["markdown"]],  # Using extraction as context
            "ground_truth": json.dumps(ground_truth)
        }
        
        ragas_results = await ragas_evaluator.evaluate_single(ragas_input)
        
        logger.info(f"\nRAGAS Scores:")
        logger.info(f"  - Faithfulness: {ragas_results.get('faithfulness', 0):.3f}")
        logger.info(f"  - Answer Relevancy: {ragas_results.get('answer_relevancy', 0):.3f}")
        logger.info(f"  - Context Precision: {ragas_results.get('context_precision', 0):.3f}")
        logger.info(f"  - Context Recall: {ragas_results.get('context_recall', 0):.3f}")
    except Exception as e:
        logger.warning(f"RAGAS evaluation error: {e}")
        ragas_results = {}
    
    # Generate comprehensive report
    logger.info("\n" + "="*80)
    logger.info("GENERATING REPORT")
    logger.info("="*80)
    
    report = generate_report(
        ground_truth=ground_truth,
        predicted_data=predicted_data,
        custom_results=custom_results,
        ragas_results=ragas_results,
        metadata=results["metadata"],
        duration=duration
    )
    
    # Save report
    report_path = output_dir / "accuracy_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"\nSaved accuracy report to: {report_path}")
    
    # Save full results JSON
    full_results = {
        "test_info": {
            "pdf": str(pdf_path),
            "firm": results['firm_detected'],
            "date": datetime.now().isoformat(),
            "duration_seconds": duration
        },
        "ground_truth_summary": {
            "pipes": len(ground_truth.get('expected_pipes', [])),
            "materials": len(ground_truth.get('expected_materials', [])),
            "volumes": len(ground_truth.get('expected_volumes', [])),
            "total_lf": total_lf
        },
        "extraction_summary": {
            "pipes": len(predicted_data.get('pipes', [])),
            "structures": len(predicted_data.get('structures', [])),
            "earthwork": len(predicted_data.get('earthwork', [])),
            "total_lf": system_lf
        },
        "custom_metrics": custom_results,
        "ragas_metrics": ragas_results
    }
    
    results_path = output_dir / "full_accuracy_results.json"
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Saved full results to: {results_path}")
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOverall Accuracy: {custom_results.get('overall_accuracy', 0):.1%}")
    logger.info(f"Processing Time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logger.info(f"\nAll results saved to: {output_dir}")
    
    return full_results


def generate_report(
    ground_truth: Dict[str, Any],
    predicted_data: Dict[str, Any],
    custom_results: Dict[str, Any],
    ragas_results: Dict[str, Any],
    metadata: Dict[str, Any],
    duration: float
) -> str:
    """Generate comprehensive accuracy report in markdown."""
    
    report = f"""# Dawn Ridge Accuracy Report
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Firm**: Hagen Engineering
**Test Duration**: {duration:.1f} seconds ({duration/60:.1f} minutes)

---

## Test Configuration

- **Model**: {metadata.get('model', 'gpt-4o')}
- **Workflow**: {"Three-pass" if metadata.get('three_pass') else "Single-pass"}
- **Pages Processed**: {metadata.get('total_pages', 'N/A')}
- **RAG Enabled**: Yes

---

## Ground Truth (from Excel Spreadsheets)

| Category | Count | Total |
|----------|-------|-------|
| Pipes | {len(ground_truth.get('expected_pipes', []))} | {sum(p.get('length_ft', 0) * p.get('count', 1) for p in ground_truth.get('expected_pipes', [])):.2f} LF |
| Materials | {len(ground_truth.get('expected_materials', []))} | - |
| Volume Items | {len(ground_truth.get('expected_volumes', []))} | - |

---

## System Extraction

| Category | Count | Total |
|----------|-------|-------|
| Pipes | {len(predicted_data.get('pipes', []))} | {sum(p.get('length_ft', 0) * p.get('count', 1) for p in predicted_data.get('pipes', [])):.2f} LF |
| Structures | {len(predicted_data.get('structures', []))} | - |
| Earthwork | {len(predicted_data.get('earthwork', []))} | - |

---

## Accuracy Metrics

### Custom Construction Metrics

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **{custom_results.get('overall_accuracy', 0):.1%}** |
| Pipe Count Accuracy | {custom_results.get('pipe_count_accuracy', 0):.1%} |
| Total LF Accuracy | {custom_results.get('total_lf_accuracy', 0):.1%} |
| Material Accuracy | {custom_results.get('material_accuracy', 0):.1%} |
| Depth Extraction Rate | {custom_results.get('depth_extraction_rate', 0):.1%} |
| Elevation Accuracy | {custom_results.get('elevation_accuracy', 0):.1%} |
| Volume Detection Rate | {custom_results.get('volume_detection_rate', 0):.1%} |

### RAGAS Metrics

| Metric | Score |
|--------|-------|
| Faithfulness | {ragas_results.get('faithfulness', 0):.3f} |
| Answer Relevancy | {ragas_results.get('answer_relevancy', 0):.3f} |
| Context Precision | {ragas_results.get('context_precision', 0):.3f} |
| Context Recall | {ragas_results.get('context_recall', 0):.3f} |

---

## Detailed Breakdown

### Pipe Analysis

**Ground Truth**: {len(ground_truth.get('expected_pipes', []))} items, {sum(p.get('length_ft', 0) * p.get('count', 1) for p in ground_truth.get('expected_pipes', [])):.2f} LF
**System Found**: {len(predicted_data.get('pipes', []))} items, {sum(p.get('length_ft', 0) * p.get('count', 1) for p in predicted_data.get('pipes', [])):.2f} LF

**Accuracy**: {custom_results.get('pipe_count_accuracy', 0):.1%}

### Material Analysis

**Ground Truth**: {len(ground_truth.get('expected_materials', []))} items
**System Found**: {len(predicted_data.get('pipes', []))} pipe items with materials

**Accuracy**: {custom_results.get('material_accuracy', 0):.1%}

### Earthwork/Grading Analysis

**Ground Truth**: {len(ground_truth.get('expected_volumes', []))} volume items
**System Found**: {len(predicted_data.get('earthwork', []))} earthwork items

**Detection Rate**: {custom_results.get('volume_detection_rate', 0):.1%}

---

## Key Findings

### Strengths
{_generate_strengths(custom_results)}

### Areas for Improvement
{_generate_improvements(custom_results)}

---

## Recommendations

{_generate_recommendations(custom_results)}

---

## Next Steps

1. Review mismatches between ground truth and system extraction
2. Enhance few-shot examples for underperforming categories
3. Iterate on prompt engineering for edge cases
4. Add additional Hagen Engineering examples from other documents

---

*Generated by EstimAI Production Accuracy Testing System*
"""
    
    return report


def _generate_strengths(results: Dict[str, Any]) -> str:
    """Generate strengths section based on results."""
    strengths = []
    
    if results.get('pipe_count_accuracy', 0) > 0.7:
        strengths.append("- Strong pipe detection and counting")
    if results.get('depth_extraction_rate', 0) > 0.7:
        strengths.append("- Effective depth measurement extraction")
    if results.get('elevation_accuracy', 0) > 0.7:
        strengths.append("- Accurate elevation and invert extraction")
    if results.get('material_accuracy', 0) > 0.7:
        strengths.append("- Reliable material identification")
    
    if not strengths:
        strengths.append("- System is functioning and producing structured output")
    
    return "\n".join(strengths)


def _generate_improvements(results: Dict[str, Any]) -> str:
    """Generate improvements section based on results."""
    improvements = []
    
    if results.get('pipe_count_accuracy', 0) < 0.7:
        improvements.append("- Pipe detection and counting needs enhancement")
    if results.get('total_lf_accuracy', 0) < 0.7:
        improvements.append("- Linear footage calculation accuracy")
    if results.get('depth_extraction_rate', 0) < 0.7:
        improvements.append("- Depth measurement extraction")
    if results.get('elevation_accuracy', 0) < 0.7:
        improvements.append("- Elevation and invert reading accuracy")
    if results.get('volume_detection_rate', 0) < 0.5:
        improvements.append("- Earthwork and volume detection")
    
    if not improvements:
        improvements.append("- Continue refinement of few-shot examples")
    
    return "\n".join(improvements)


def _generate_recommendations(results: Dict[str, Any]) -> str:
    """Generate recommendations based on results."""
    overall = results.get('overall_accuracy', 0)
    
    if overall > 0.8:
        return """
The system demonstrates strong performance on Hagen Engineering documents. 
Recommendations:
1. Test on additional Hagen documents to validate consistency
2. Begin gathering examples from second engineering firm
3. Deploy to limited production testing
"""
    elif overall > 0.6:
        return """
The system shows promising results but needs refinement.
Recommendations:
1. Enhance few-shot examples for underperforming categories
2. Review and improve prompt engineering
3. Analyze specific failure cases
4. Add more Hagen-specific notation patterns
"""
    else:
        return """
The system requires significant improvement.
Recommendations:
1. Conduct detailed error analysis
2. Review and revise prompt templates
3. Expand few-shot examples significantly
4. Consider adjusting three-pass workflow parameters
5. Validate RAG retrieval quality
"""


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    asyncio.run(run_accuracy_test())

