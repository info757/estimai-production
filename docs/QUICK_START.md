# Quick Start Guide

## Prerequisites

- Python 3.11+
- OpenAI API key
- Qdrant (included in repo)

## Installation

### 1. Create Virtual Environment

```bash
cd /Users/williamholt/estimai-production
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: `pdf2image` requires `poppler`. Install it:

**macOS**:
```bash
brew install poppler
```

**Ubuntu/Debian**:
```bash
sudo apt-get install poppler-utils
```

**Windows**:
Download from: http://blog.alivate.com.au/poppler-windows/

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 4. Verify Setup

```bash
python -c "import app.vision; print('✅ Setup successful!')"
```

## Run Accuracy Test

Test the system on Dawn Ridge (Hagen Engineering):

```bash
python scripts/run_accuracy_test.py
```

This will:
1. Analyze all 25 pages of the Dawn Ridge PDF
2. Extract pipes, structures, and earthwork
3. Compare against ground truth from Excel spreadsheets
4. Generate accuracy report
5. Save results to `results/` directory

**Expected Duration**: 5-10 minutes for 25 pages

## Results

After running the test, check:

- `results/dawn_ridge_extraction.md` - Raw markdown extraction
- `results/dawn_ridge_extraction.json` - Parsed JSON data
- `results/accuracy_report.md` - Comprehensive accuracy report
- `results/full_accuracy_results.json` - Detailed metrics

## Using the System

### Python API

```python
import asyncio
from app.vision import UniversalVisionAgent, parse_markdown_to_json

async def extract_document():
    # Initialize agent
    agent = UniversalVisionAgent()
    
    # Analyze document
    results = await agent.analyze_document(
        pdf_path="data/test_pdfs/Dawn Ridge Homes_HEPA_Combined_04-1-25.pdf",
        firm="hagen_engineering",  # or auto-detect
        use_three_pass=True
    )
    
    # Get markdown
    markdown = results["markdown"]
    
    # Parse to JSON
    structured_data = parse_markdown_to_json(markdown)
    
    return structured_data

# Run
data = asyncio.run(extract_document())
print(f"Found {len(data['pipes'])} pipes")
```

### Command Line

```bash
python -m app.vision.universal_agent --pdf path/to/document.pdf --firm hagen_engineering
```

## Adding New Engineering Firm

When you encounter documents from a new firm:

1. Extract notation patterns from their documents
2. Create few-shot examples (see `prompts/firm_specific_examples.py`)
3. Add to `FIRM_EXAMPLES` dictionary
4. Test on their documents

Example structure:

```python
"new_firm_engineering": {
    "firm_name": "New Firm Engineering",
    "detection_keywords": ["NEW FIRM", "NEW FIRM ENGINEERING"],
    "notation_guide": {
        "sanitary": ["SS", "SAN"],
        # ... more abbreviations ...
    },
    "mainline_pipes": [
        {
            "description": "Example pipe from their documents",
            "visual_notation": "8\" PVC SS",
            "markdown_output": """..."""
        }
    ],
    # ... more examples ...
}
```

## Troubleshooting

### Import Errors

If you see import errors:
```bash
export PYTHONPATH="/Users/williamholt/estimai-production:$PYTHONPATH"
```

### PDF Processing Errors

If `pdf2image` fails:
- Verify poppler is installed: `which pdftoppm`
- Check PDF is not corrupted: `pdfinfo your-file.pdf`

### OpenAI API Errors

- Verify API key is set: `echo $OPENAI_API_KEY`
- Check API quota and limits
- Model `gpt-4o` requires access (fallback: `gpt-4-vision-preview`)

### Qdrant Connection Issues

Qdrant storage is included in `data/qdrant_storage/`. The system accesses it locally - no Docker needed.

## Performance Tuning

### Speed vs. Quality Trade-offs

**Faster** (use for testing):
```python
agent = UniversalVisionAgent(
    model="gpt-4o-mini",  # Faster, cheaper
    temperature=0.1
)
results = await agent.analyze_document(
    pdf_path=pdf_path,
    use_three_pass=False  # Single-pass
)
```

**Higher Quality** (use for production):
```python
agent = UniversalVisionAgent(
    model="gpt-4o",  # Best vision model
    temperature=0.05  # Very deterministic
)
results = await agent.analyze_document(
    pdf_path=pdf_path,
    use_three_pass=True  # Context preservation
)
```

### Parallel Processing

For multi-document processing:

```python
import asyncio

async def process_multiple():
    agent = UniversalVisionAgent()
    
    pdfs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    
    tasks = [
        agent.analyze_document(pdf, firm="hagen_engineering")
        for pdf in pdfs
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

## Next Steps

1. ✅ Run accuracy test on Dawn Ridge
2. Review accuracy report
3. If accuracy > 80%, test on additional Hagen documents
4. If accuracy < 80%, iterate on prompts and examples
5. Once stable, add second engineering firm

## Support

For issues or questions:
- Check `docs/` directory for detailed architecture
- Review `prompts/firm_specific_examples.py` for example patterns
- Examine `app/vision/universal_agent.py` for implementation details

