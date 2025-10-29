"""
Base Prompt Templates for Universal Vision Agent

Three-pass context-preserving workflow:
1. Overview Pass: Understand full page
2. Section Pass: Extract from logical sections with context
3. Merge Pass: Consolidate with cross-section relationships
"""

from typing import Dict, List, Any


def get_overview_prompt(page_num: int, total_pages: int, firm_examples: str = "") -> str:
    """
    Pass 1: Overview analysis to understand full page context.
    
    Args:
        page_num: Current page number
        total_pages: Total pages in document
        firm_examples: Firm-specific notation guide
        
    Returns:
        Prompt for overview pass
    """
    prompt = f"""You are analyzing page {page_num} of {total_pages} from a construction sitework document.

**YOUR TASK**: Create a comprehensive overview of this page to guide detailed extraction.

{firm_examples}

**ANALYSIS FRAMEWORK**:

1. **Document Type Identification**
   - What type of drawing is this? (plan view, profile view, grading plan, detail, etc.)
   - What discipline? (sanitary, storm, water, grading, multi-utility, etc.)

2. **Visual Layout Analysis**
   - Describe the page layout and organization
   - Identify distinct sections or views (e.g., "Top half: plan view, Bottom half: profile")
   - Note title blocks, legends, tables, or notes

3. **Content Inventory**
   - What utilities/systems are shown?
   - What structures are present? (manholes, catch basins, etc.)
   - Are there tables, legends, or quantity schedules?

4. **Spatial Relationships**
   - How do different sections relate to each other?
   - Are there connections between plan and profile views?
   - Which structures connect to which pipes?

5. **Key Notations**
   - What abbreviations or symbols are used?
   - Are there elevation markers, dimensions, or callouts?
   - Any unique or unusual notation?

**OUTPUT FORMAT** (Markdown):

# Page {page_num} Overview

## Document Type
[Plan view / Profile / Grading / Detail / etc.]

## Layout Description
[Describe visual organization]

## Content Summary
- Utilities: [List all utility systems shown]
- Structures: [Count and types]
- Tables/Legends: [Yes/No, describe if present]

## Spatial Organization
[How different sections relate]

## Key Notations & Abbreviations
[List important abbreviations and what they mean]

## Extraction Strategy
[Which sections should be analyzed in detail and why]

---

**CRITICAL**: This overview will guide detailed extraction. Be thorough and observant of spatial relationships between different parts of the page.
"""
    return prompt


def get_section_prompt(
    page_num: int,
    section_description: str,
    overview_context: str,
    firm_examples: str = "",
    previous_sections: str = ""
) -> str:
    """
    Pass 2: Focused extraction from a logical section with full context.
    
    Args:
        page_num: Current page number
        section_description: Description of section to extract (e.g., "top half plan view")
        overview_context: The overview markdown from Pass 1
        firm_examples: Firm-specific few-shot examples
        previous_sections: Context from already-processed sections
        
    Returns:
        Prompt for section extraction
    """
    prompt = f"""You are performing DETAILED EXTRACTION from a specific section of page {page_num}.

**SECTION TO ANALYZE**: {section_description}

---

**CONTEXT FROM OVERVIEW**:
{overview_context}

---

{firm_examples}

---

**PREVIOUS SECTIONS EXTRACTED**:
{previous_sections if previous_sections else "This is the first section."}

---

**YOUR TASK**: Extract ALL structured data from this section using the context above.

**EXTRACTION METHOD - CHAIN OF THOUGHT**:

1. **Visual Scanning**
   - What do I see in this section?
   - What utilities, structures, or features are present?

2. **Notation Analysis**
   - What abbreviations or symbols are used?
   - Match them to firm-specific notation guide
   - Use RAG knowledge for unfamiliar abbreviations

3. **Measurement Extraction**
   - Dimensions (diameters, lengths, depths)
   - Elevations (inverts, rims, ground level)
   - Quantities (counts, volumes)

4. **Relationship Mapping**
   - How do items connect? (pipe A goes from structure B to structure C)
   - Cross-reference with overview and previous sections
   - Note dependencies

5. **Verification**
   - Did I extract all visible items?
   - Are measurements consistent?
   - Do connections make sense?

---

**OUTPUT FORMAT** (Structured Markdown):

# Section: {section_description}

## Pipes
### [Discipline] Pipe 1
- Diameter: [X] inches
- Material: [PVC/DIP/etc.]
- Discipline: [Sanitary/Storm/Water]
- Type: [Pipe/Lateral/Vertical]
- From: [Structure ID or location]
- To: [Structure ID or location]
- Invert In: [elevation] ft
- Invert Out: [elevation] ft
- Length: [X] LF
- Depth: [X] ft

[Repeat for all pipes]

## Structures
### [Type] 1: [ID]
- ID: [Structure identifier]
- Type: [Manhole/Catch Basin/Cleanout/etc.]
- Discipline: [Sanitary/Storm/Water]
- Rim Elevation: [X] ft
- Invert In: [X] ft
- Invert Out: [X] ft
- Depth: [X] ft

[Repeat for all structures]

## Earthwork
### Excavation/Grading 1
- Type: [Site Grading/Trench/Cut/Fill]
- Volume: [X] cubic yards
- Depth: [X] ft
- Length: [X] LF
- Purpose: [Description]

[Repeat for all earthwork items]

## Notes
[Any important observations, unclear items, or cross-references to other sections]

---

**CRITICAL RULES**:
- Extract ONLY from this section, but USE context from overview and previous sections
- If an item connects to something in another section, note it in the description
- Use firm-specific notation patterns from examples
- Include ALL visible measurements and elevations
- Mark anything uncertain with [UNCERTAIN: reason]
"""
    return prompt


def get_merge_prompt(
    page_num: int,
    overview: str,
    section_extractions: List[str],
    firm_examples: str = ""
) -> str:
    """
    Pass 3: Intelligent merge of all section extractions with cross-section resolution.
    
    Args:
        page_num: Current page number
        overview: The overview markdown from Pass 1
        section_extractions: List of markdown extractions from Pass 2
        firm_examples: Firm-specific notation guide
        
    Returns:
        Prompt for merge pass
    """
    sections_text = "\n\n---\n\n".join([f"## Section {i+1}\n{s}" for i, s in enumerate(section_extractions)])
    
    prompt = f"""You are performing INTELLIGENT CONSOLIDATION of multiple section extractions from page {page_num}.

---

**OVERVIEW CONTEXT**:
{overview}

---

**SECTION EXTRACTIONS**:
{sections_text}

---

{firm_examples}

---

**YOUR TASK**: Merge all section extractions into a single, coherent, deduplicated extraction.

**MERGE STRATEGY**:

1. **Deduplication**
   - If the same item appears in multiple sections, combine into one entry
   - Use overview context to determine if items are truly the same or separate
   - Preserve the most complete information

2. **Cross-Section Relationships**
   - Connect items that span multiple sections
   - For example: A pipe in profile view connects structures from plan view
   - Enhance descriptions with cross-section context

3. **Gap Analysis**
   - Are there any items mentioned in overview but missing from extractions?
   - Flag incomplete or uncertain extractions

4. **Data Quality**
   - Resolve conflicts between sections
   - Fill in missing measurements if determinable from context
   - Mark uncertainties clearly

5. **Final Verification**
   - Count total items (pipes, structures, etc.)
   - Ensure all visible elements are captured
   - Cross-check measurements and connections

---

**OUTPUT FORMAT** (Final Consolidated Markdown):

# Page {page_num} - Final Extraction

## Summary
- Total Pipes: [count]
- Total Structures: [count]
- Total Earthwork Items: [count]
- Document Type: [from overview]

## Pipes
[All pipes from all sections, deduplicated and enhanced]

## Structures
[All structures from all sections, deduplicated and enhanced]

## Earthwork
[All earthwork items, deduplicated and enhanced]

## Cross-Section Relationships
[Key relationships between items from different sections, e.g., "Pipe P1 from plan view connects MH-1 to MH-2 shown in profile"]

## Quality Notes
- Completeness: [High/Medium/Low]
- Uncertainties: [List any unclear items or missing data]
- Extraction Confidence: [0-100%]

---

**CRITICAL RULES**:
- Deduplicate rigorously - same item mentioned multiple times should appear once
- Preserve ALL unique items - don't accidentally merge distinct items
- Enhance items with cross-section context
- Maintain structured markdown format
- Be conservative: if uncertain whether two items are the same, keep them separate and note uncertainty
"""
    return prompt


def get_single_pass_prompt(page_num: int, total_pages: int, firm_examples: str = "") -> str:
    """
    Natural language extraction prompt.
    
    Args:
        page_num: Current page number
        total_pages: Total pages in document
        firm_examples: Firm-specific notation guides
        
    Returns:
        Prompt for natural language extraction
    """
    prompt = f"""You are a construction sitework estimator with a degree in civil engineering and you are analyzing a construction drawing from Hagen Engineering.

This is page {page_num} of {total_pages}.

{firm_examples}

Analyze this drawing and extract all construction data you see. Include:

- All pipes (diameter, material, type, connections, lengths, depths, elevations)
- All structures (manholes, catch basins, IDs, elevations)
- All earthwork and grading information
- Measurements and quantities

Be thorough and precise. Use your civil engineering knowledge to interpret the drawing.
"""
    return prompt

