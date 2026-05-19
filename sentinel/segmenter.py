from __future__ import annotations
import re
from typing import List
from sentinel.models import Clause


class ClauseSegmenter:
    """Protocol-like interface for clause segmentation."""

    def segment(self, text: str, source_input_id: str) -> List[Clause]:
        ...


class SpacyClauseSegmenter(ClauseSegmenter):
    """Robust clause segmentation with spaCy + dependent clause detection."""

    # Relative/subordinate clause patterns that indicate dependent clauses
    DEPENDENT_PATTERNS = [
        r'\bwhich\b',   # "text which..." (non-restrictive relative)
        r'\bthat\b',    # "text that..." (dependent clause after conjunction/punct)
        r'\bas\b',      # "text as..." (comparison/relative)
        r'\bwhere\b',   # "text where..." (relative location)
        r'\bwhile\b',   # can be contrastive or relative
    ]

    # Subordination/conjunction markers within clauses
    SUBORDINATION_MARKERS = ['however', 'though', 'although', 'because', 'since', 'if', 'unless']

    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            print("Using spaCy segmenter")
        except (OSError, ImportError) as e:
            print(f"spaCy unavailable ({e}). Using regex fallback.")
            self.nlp = None
    
    def segment(self, text: str, source_input_id: str) -> List[Clause]:
        if self.nlp is None:
            return _fallback_segment(text, source_input_id)
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        clauses = []
        
        for idx, sent_text in enumerate(sentences):
            sub_units = _split_clause_units_with_dependent(self, sent_text)
            
            for pos, unit in enumerate(sub_units):
                if unit.strip():
                    clauses.append(Clause(
                        id=f"{source_input_id}:{idx}#{pos}",
                        text=unit.strip(),
                        position=len(clauses),
                        source_input_id=source_input_id
                    ))
        
        return clauses


def _split_clause_units_with_dependent(segmenter, sentence: str) -> List[str]:
    """Split a sentence into clause-level sub-units including dependent clauses."""
    units = []
    
    # Step 1: Remove trailing period for processing (we'll add it back)
    processed = sentence.rstrip('.')
    
    # Step 2: Handle discourse markers first (these start new independent clauses)
    parts_by_discourse = re.split(
        r'\b(?:however|therefore|consequently|otherwise|although|though)\s+',
        processed, flags=re.IGNORECASE
    )
    
    for i, part in enumerate(parts_by_discourse):
        if i == 0:
            # First part has no discourse marker prefix
            part = ' '.join(part.split())
        else:
            # Discourse-marked parts get preserved with their markers
            part = f"{re.sub(r's+', ' ', part).strip()}"
        
        if not part.strip():
            continue
        
        # Step 3: Look for relative clauses within the current part
        # Pattern: text, which/that + clause
        # Example: "I feel happy but I am nervous. However, the test is tomorrow, which will determine my grade."
        
        unit = _extract_relative_clauses(part)
        
        if not isinstance(unit, list):
            unit = [unit] if unit else []
        
        for clause in unit:
            # Step 4: Split by conjunctions within clauses (but/and/or)
            sub_parts = re.split(r'\b(?:but|and|or)\b', clause, flags=re.IGNORECASE)
            
            for sub in sub_parts:
                sub = sub.strip()
                
                # Skip single words or fragments
                words = len(sub.split())
                if words >= 2 or '.' in sub:
                    units.append(sub)
    
    return units


def _extract_relative_clauses(text: str) -> List[str]:
    """Extract relative/subordinate clauses marked by commas and relative pronouns."""
    units = []
    
    # Pattern matches text that ends with comma + relative pronoun (which, that, where, as)
    relative_pattern = r',\s*(?:which|that|where|as)\b'
    
    # Find splits at relative clause boundaries
    parts = re.split(relative_pattern, text)
    
    for i, part in enumerate(parts):
        part = part.strip()
        
        if not part:
            continue
        
        # If this is the first part (no relative pronoun at start), handle it normally
        if i == 0:
            # Check if the part contains a relative clause embedded within
            embedded_clause = _find_embedded_relative_clause(part)
            
            if embedded_clause:
                # Split around embedded relative clause
                before, after = embedded_clause.split(',', maxsplit=1)
                
                before = before.strip()
                after = ', ' + after.strip()  # Re-add comma and space
                
                if before:
                    units.append(before)
                units.append(after)
            else:
                units.append(part)
        else:
            # This part starts with relative pronoun — it's a dependent clause
            units.append(','.strip() + ' ' + part.strip()) if ',' in ','.strip() else part
    
    return units


def _find_embedded_relative_clause(text: str) -> str | None:
    """Find an embedded relative clause pattern (text, which/that...)."""
    # Look for ", which" or ", that" patterns within the text
    matches = re.findall(r'([^\,]+),\s*(which|that)\b', text, flags=re.IGNORECASE)
    
    if not matches:
        return None
    
    # If multiple matches, find the last one (most likely to be a dependent clause at end)
    last_match_text = ''
    for match in reversed(matches):
        before_clause = match[0]  # text before relative pronoun
        after_clause = match[1]   # relative pronoun
        
        # Check if this looks like an end-of-sentence dependent clause pattern
        after_words = len(after_clause.split())
        if after_words >= 2:
            last_match_text = f"{before_clause}, {after_clause}"
            break
    
    if last_match_text:
        return last_match_text
    return None


def _fallback_segment(text: str, source_input_id: str) -> List[Clause]:
    """Regex-based fallback when spaCy is unavailable."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    units = []
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        
        sentence = sentence.lstrip('.')
        
        # Handle relative clause patterns in fallback mode too
        parts = re.split(r',\s*(which|that)\b', sentence, flags=re.IGNORECASE)
        
        for i, part in enumerate(parts):
            part = part.strip()
            
            if not part:
                continue
            
            # Split by conjunctions
            sub_parts = re.split(r'\b(?:but|and|or)\b', part, flags=re.IGNORECASE)
            
            for sub in sub_parts:
                sub = sub.strip()
                
                # Skip very short fragments
                words = len(sub.split())
                if words >= 2 or '.' in sub:
                    units.append(sub)
    
    return [
        Clause(id=f"{source_input_id}:{index}", text=unit, position=index, source_input_id=source_input_id)
        for index, unit in enumerate(units)
    ]


def create_simple_segmenter() -> SpacyClauseSegmenter:
    """Factory function to ensure consistent segmenter creation."""
    return SpacyClauseSegmenter()
