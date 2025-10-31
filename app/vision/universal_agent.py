"""
Universal Vision Agent for Construction Document Extraction

Single comprehensive agent with:
- Markdown output (not rigid JSON)
- Three-pass context preservation
- Firm-specific few-shot learning
- RAG integration for abbreviations
- Spatial section decomposition
"""

import os
import base64
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pdf2image import convert_from_path
from PIL import Image
import io
import fitz  # PyMuPDF for precise region cropping

# Import RAG and prompts
from app.rag.advanced_retriever import AdvancedRetriever
from prompts.firm_specific_examples import (
    detect_firm_from_page,
    format_examples_for_prompt,
    get_notation_guide,
    FIRM_EXAMPLES
)
from prompts.base_prompts import (
    get_overview_prompt,
    get_section_prompt,
    get_merge_prompt,
    get_single_pass_prompt
)

logger = logging.getLogger(__name__)


class UniversalVisionAgent:
    """
    Single comprehensive vision agent for construction document analysis.
    
    Uses three-pass workflow:
    1. Overview: Understand full page context
    2. Section Extraction: Process logical spatial sections
    3. Merge: Consolidate with cross-section relationships
    """
    
    def __init__(
        self,
        rag_retriever: Optional[AdvancedRetriever] = None,
        model: str = "gpt-4o",
        temperature: float = 0.1
    ):
        """
        Initialize Universal Vision Agent.
        
        Args:
            rag_retriever: Advanced RAG retriever for construction knowledge
            model: Vision model to use (gpt-4o recommended)
            temperature: LLM temperature (low for precision)
        """
        self.model = model
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature
        )
        
        # Initialize RAG retriever
        self.rag = rag_retriever or AdvancedRetriever()
        
        logger.info(f"Universal Vision Agent initialized with model={model}")
    
    async def analyze_document(
        self,
        pdf_path: str,
        firm: str = "hagen_engineering",
        auto_detect_firm: bool = True,
        use_three_pass: bool = True,
        page_range: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Analyze construction document and extract structured data.
        
        Args:
            pdf_path: Path to PDF document
            firm: Firm identifier (e.g., "hagen_engineering")
            auto_detect_firm: Auto-detect firm from page content
            use_three_pass: Use three-pass workflow (vs single-pass)
            page_range: Optional list of page numbers to process (1-indexed)
            
        Returns:
            Dictionary with:
            - "markdown": Consolidated markdown extraction
            - "pages": List of per-page markdown extractions
            - "firm_detected": Detected firm name
            - "metadata": Document metadata
        """
        logger.info(f"Analyzing document: {pdf_path}")
        logger.info(f"Initial firm: {firm}, auto_detect={auto_detect_firm}, three_pass={use_three_pass}")
        
        # Convert PDF to images
        pages_data = await self._load_pdf_pages(pdf_path, page_range)
        total_pages = len(pages_data)
        
        logger.info(f"Loaded {total_pages} pages")
        
        # Detect firm if requested
        if auto_detect_firm:
            detected_firm = await self._detect_firm_from_first_page(pages_data[0])
            if detected_firm != "generic":
                firm = detected_firm
                logger.info(f"Auto-detected firm: {firm}")
        
        # Get firm-specific examples
        firm_examples = format_examples_for_prompt(firm)
        
        # Process each page
        page_results = []
        for i, page_data in enumerate(pages_data):
            page_num = page_data["page_num"]
            logger.info(f"Processing page {page_num}/{total_pages}")
            
            if use_three_pass:
                page_markdown = await self._three_pass_extraction(
                    page_data=page_data,
                    page_num=page_num,
                    total_pages=total_pages,
                    firm_examples=firm_examples,
                    firm=firm
                )
            else:
                page_markdown = await self._single_pass_extraction(
                    page_data=page_data,
                    page_num=page_num,
                    total_pages=total_pages,
                    firm_examples=firm_examples,
                    firm=firm
                )
            
            page_results.append({
                "page_num": page_num,
                "markdown": page_markdown
            })
        
        # Consolidate all pages
        consolidated_markdown = self._consolidate_pages(page_results, firm)
        
        return {
            "markdown": consolidated_markdown,
            "pages": page_results,
            "firm_detected": firm,
            "metadata": {
                "total_pages": total_pages,
                "model": self.model,
                "three_pass": use_three_pass
            }
        }
    
    def _crop_pdf_region(
        self,
        pdf_path: str,
        page_num: int,
        bbox: tuple,
        zoom_dpi: int = 300
    ) -> Optional[str]:
        """
        Crop a specific region from a PDF page and return as base64 image.
        Since PDFs are vectorized, zooming maintains clarity.
        
        Args:
            pdf_path: Path to PDF
            page_num: 1-indexed page number
            bbox: (x0, y0, x1, y1) bounding box in PDF points (72 DPI)
            zoom_dpi: DPI to render at (higher = more zoom/clarity)
            
        Returns:
            Base64-encoded image string or None
        """
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num - 1)  # 0-indexed
            
            # Create a matrix to render at higher DPI
            zoom = zoom_dpi / 72.0  # Convert DPI to zoom factor
            mat = fitz.Matrix(zoom, zoom)
            
            # Adjust bbox for zoom
            rect = fitz.Rect(bbox)
            
            # Render the region
            pix = page.get_pixmap(matrix=mat, clip=rect)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            doc.close()
            return image_b64
        except Exception as e:
            logger.warning(f"Failed to crop PDF region: {e}")
            return None

    async def _load_pdf_pages(
        self,
        pdf_path: str,
        page_range: Optional[List[int]] = None,
        dpi: int = 150
    ) -> List[Dict[str, Any]]:
        """
        Load PDF pages as images.
        
        Args:
            pdf_path: Path to PDF
            page_range: Optional list of page numbers (1-indexed)
            
        Returns:
            List of dicts with page_num and image_b64
        """
        pages = convert_from_path(pdf_path, dpi=dpi)
        
        pages_data = []
        for i, page_img in enumerate(pages):
            page_num = i + 1
            
            # Filter by page_range if provided
            if page_range and page_num not in page_range:
                continue
            
            # Convert PIL image to base64
            buffer = io.BytesIO()
            page_img.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            pages_data.append({
                "page_num": page_num,
                "image_b64": image_b64,
                "pdf_path": pdf_path  # Store for later high-DPI reloads
            })
        
        return pages_data
    
    async def _detect_firm_from_first_page(self, page_data: Dict[str, Any]) -> str:
        """
        Detect engineering firm from first page (title block).
        
        Args:
            page_data: Page data with image_b64
            
        Returns:
            Firm identifier
        """
        # Use vision to extract text from title block
        prompt = """Look at this construction drawing page. 
        
Extract the engineering firm name from the title block (usually bottom right corner).

Return ONLY the firm name, nothing else."""
        
        try:
            response = await self._call_vision_llm(
                image_b64=page_data["image_b64"],
                system_prompt="You extract text from images.",
                user_prompt=prompt
            )
            
            firm = detect_firm_from_page(response)
            return firm
        except Exception as e:
            logger.warning(f"Error detecting firm: {e}")
            return "generic"
    
    async def _three_pass_extraction(
        self,
        page_data: Dict[str, Any],
        page_num: int,
        total_pages: int,
        firm_examples: str,
        firm: str
    ) -> str:
        """
        Three-pass context-preserving extraction.
        
        Args:
            page_data: Page data with image
            page_num: Current page number
            total_pages: Total pages
            firm_examples: Formatted firm examples
            firm_name: Firm identifier
            
        Returns:
            Consolidated markdown for page
        """
        # Pass 1: Overview
        logger.info(f"  Pass 1: Overview analysis")
        overview_prompt = get_overview_prompt(page_num, total_pages, firm_examples)
        overview = await self._call_vision_llm(
            image_b64=page_data["image_b64"],
            system_prompt="You are a construction document analyst creating page overviews.",
            user_prompt=overview_prompt
        )
        
        # Determine sections from overview
        sections = self._determine_sections_from_overview(overview)
        logger.info(f"  Identified {len(sections)} sections: {sections}")
        
        # Pass 2: Section extractions
        section_extractions = []
        pdf_path = page_data.get("pdf_path")  # Need to pass this through
        
        for section_desc in sections:
            logger.info(f"  Pass 2: Extracting {section_desc}")
            
            # Determine if this section needs higher resolution
            # Profile sections, callouts, and detailed views benefit from zoom
            # Also check for "grading" which often contains profile views
            section_lower = section_desc.lower()
            needs_zoom = any(keyword in section_lower for keyword in [
                "profile", "callout", "detail", "legend", "table", "note", "grading"
            ])
            
            # Profile sections need the highest resolution for reading callouts
            is_profile = "profile" in section_lower
            
            # Use higher DPI image for sections that need zoom (vector PDFs stay sharp)
            section_image_b64 = page_data["image_b64"]
            if needs_zoom:
                # Profile sections need highest resolution to read small callout text like "DIP", "PVC"
                zoom_dpi = 500 if is_profile else 400
                logger.info(f"  Using higher resolution ({zoom_dpi} DPI) for better text clarity")
                # Reload page at higher DPI for this section
                high_dpi_pages = convert_from_path(pdf_path, dpi=zoom_dpi, first_page=page_num, last_page=page_num)
                if high_dpi_pages:
                    buffer = io.BytesIO()
                    high_dpi_pages[0].save(buffer, format="PNG")
                    section_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Get RAG context for this section
            rag_context = await self._get_rag_context(section_desc, firm)
            
            # Build section prompt with all context
            section_prompt = get_section_prompt(
                page_num=page_num,
                section_description=section_desc,
                overview_context=overview,
                firm_examples=firm_examples + "\n\n" + rag_context,
                previous_sections="\n\n".join(section_extractions)
            )
            
            section_markdown = await self._call_vision_llm(
                image_b64=section_image_b64,
                system_prompt="You are a construction data extractor analyzing specific sections.",
                user_prompt=section_prompt
            )
            
            section_extractions.append(section_markdown)
        
        # Pass 3: Merge
        logger.info(f"  Pass 3: Merging {len(section_extractions)} sections")
        merge_prompt = get_merge_prompt(
            page_num=page_num,
            overview=overview,
            section_extractions=section_extractions,
            firm_examples=firm_examples
        )
        
        # Merge doesn't need image, use text-only LLM
        merged = await self._call_text_llm(
            system_prompt="You are an expert at consolidating construction data extractions.",
            user_prompt=merge_prompt
        )
        
        return merged
    
    async def _single_pass_extraction(
        self,
        page_data: Dict[str, Any],
        page_num: int,
        total_pages: int,
        firm_examples: str,
        firm: str
    ) -> str:
        """
        Single-pass extraction for simpler pages.
        
        Args:
            page_data: Page data with image
            page_num: Current page number
            total_pages: Total pages
            firm_examples: Formatted firm examples
            firm: Firm identifier
            
        Returns:
            Markdown extraction
        """
        logger.info(f"  Single-pass extraction")
        
        # Get RAG context
        rag_context = await self._get_rag_context(f"construction page {page_num}", firm)
        
        # Build prompt
        prompt = get_single_pass_prompt(page_num, total_pages, firm_examples + "\n\n" + rag_context)
        
        # Extract
        markdown = await self._call_vision_llm(
            image_b64=page_data["image_b64"],
            system_prompt="You are a construction document analyzer extracting structured data.",
            user_prompt=prompt
        )
        
        return markdown
    
    def _determine_sections_from_overview(self, overview: str) -> List[str]:
        """
        Parse overview to determine logical sections to extract.
        
        Args:
            overview: Overview markdown
            
        Returns:
            List of section descriptions
        """
        # Simple heuristic: Look for section mentions
        overview_lower = overview.lower()
        
        sections = []
        
        # Check for common section types
        if "plan view" in overview_lower or "plan" in overview_lower:
            sections.append("Plan view (top section)")
        
        if "profile" in overview_lower or "profile view" in overview_lower:
            sections.append("Profile view (bottom section)")
        
        if "grading" in overview_lower or "grading plan" in overview_lower:
            sections.append("Grading plan")
        
        if "detail" in overview_lower:
            sections.append("Detail section")
        
        if "legend" in overview_lower or "table" in overview_lower:
            sections.append("Legend/table section")
        
        # If no sections identified, treat as single section
        if not sections:
            sections = ["Full page"]
        
        return sections
    
    async def _get_rag_context(self, query: str, firm: str) -> str:
        """
        Get RAG context for abbreviations and standards.
        
        Args:
            query: Query for RAG
            firm: Firm identifier for context
            
        Returns:
            Formatted RAG context string
        """
        try:
            # Get notation guide
            notation = get_notation_guide(firm)
            
            # Query RAG
            rag_results = self.rag.retrieve_with_expansion(
                query=query + " " + " ".join(notation.keys())[:100],
                k=5
            )
            
            # Format RAG results
            rag_text = "## Construction Standards (RAG):\n"
            for i, doc in enumerate(rag_results[:3], 1):
                # RAG returns dicts, not document objects
                content = doc.get('page_content', str(doc))[:200]
                rag_text += f"{i}. {content}...\n"
            
            return rag_text
        except Exception as e:
            logger.warning(f"RAG context error: {e}")
            return ""
    
    async def _call_vision_llm(
        self,
        image_b64: Union[str, List[str]],
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Call vision LLM with image and prompts.
        
        Args:
            image_b64: Base64-encoded image OR list of images (for few-shot exemplars)
            system_prompt: System instruction
            user_prompt: User query
            
        Returns:
            LLM response text
        """
        try:
            # Normalize images input to a list
            images: List[str] = (
                [image_b64] if isinstance(image_b64, str) else list(image_b64)
            )

            # Build multimodal content: zero or more images, then text
            content: List[dict] = []
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img}",
                        "detail": "high"
                    }
                })
            content.append({
                "type": "text",
                "text": user_prompt
            })

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Vision LLM error: {e}")
            return f"Error: {str(e)}"
    
    async def _call_text_llm(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Call text-only LLM (for merge pass).
        
        Args:
            system_prompt: System instruction
            user_prompt: User query
            
        Returns:
            LLM response text
        """
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Text LLM error: {e}")
            return f"Error: {str(e)}"
    
    def _consolidate_pages(
        self,
        page_results: List[Dict[str, Any]],
        firm: str
    ) -> str:
        """
        Consolidate all page extractions into final document markdown.
        
        Args:
            page_results: List of page extraction dicts
            firm: Firm identifier
            
        Returns:
            Consolidated markdown
        """
        import datetime
        
        firm_name = FIRM_EXAMPLES.get(firm, {}).get("firm_name", "Unknown Firm")
        
        markdown = f"""# Construction Document Extraction

**Firm**: {firm_name}
**Total Pages**: {len(page_results)}
**Extraction Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}

---

"""
        
        for page_data in page_results:
            markdown += f"\n\n# Page {page_data['page_num']}\n\n"
            markdown += page_data["markdown"]
            markdown += "\n\n---\n"
        
        return markdown


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def extract_from_pdf(
    pdf_path: str,
    firm: str = "hagen_engineering",
    output_path: Optional[str] = None
) -> str:
    """
    Convenience function to extract from PDF and optionally save markdown.
    
    Args:
        pdf_path: Path to PDF
        firm: Firm identifier
        output_path: Optional path to save markdown output
        
    Returns:
        Consolidated markdown string
    """
    agent = UniversalVisionAgent()
    
    results = await agent.analyze_document(
        pdf_path=pdf_path,
        firm=firm,
        auto_detect_firm=True,
        use_three_pass=True
    )
    
    markdown = results["markdown"]
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(markdown)
        logger.info(f"Saved markdown to: {output_path}")
    
    return markdown

