"""Prompt library - Firm-specific few-shot examples and templates"""

from prompts.firm_specific_examples import FIRM_EXAMPLES, get_firm_examples, detect_firm_from_page, format_examples_for_prompt
from prompts.base_prompts import get_overview_prompt, get_section_prompt, get_merge_prompt

__all__ = [
    "FIRM_EXAMPLES",
    "get_firm_examples", 
    "detect_firm_from_page",
    "format_examples_for_prompt",
    "get_overview_prompt",
    "get_section_prompt",
    "get_merge_prompt"
]

