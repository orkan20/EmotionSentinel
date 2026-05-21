from __future__ import annotations

import re
from typing import List, Optional

from sentinel.models import Clause


class ClauseSegmenter:
    """Protocol-like interface for clause segmentation."""

    def segment(self, text: str, source_input_id: str) -> List[Clause]:
        ...


# Dependency labels whose subtree is its own clause distinct from its parent.
_CLAUSE_DEPS = frozenset({"conj", "advcl", "relcl", "ccomp"})
# A clause head must be a verbal token; noun-coord conjuncts ("salt and pepper")
# are deliberately not split.
_CLAUSE_HEAD_POS = frozenset({"VERB", "AUX"})

_KIND_FROM_DEP = {
    "ROOT": "root",
    "conj": "coordinate",
    "advcl": "subordinate",
    "relcl": "relative",
    "ccomp": "complement",
}

_SEPARATOR_PUNCT = frozenset({",", ";", ":", "."})
_TIGHT_PUNCT = frozenset({".", ",", ";", ":", "?", "!"})
_WH_TAGS = frozenset({"WDT", "WP", "WRB"})


class SpacyClauseSegmenter(ClauseSegmenter):
    """Dependency-parse clause segmenter built on spaCy.

    Each sentence is parsed; the root verb plus any verbal token with
    dependency in {conj, advcl, relcl, ccomp} becomes a clause head. A
    clause is that head's subtree minus the subtrees of any nested clause
    heads. For coordinated clauses, the connecting `cc` token (but/and/or)
    is attached so the depth scorer can see the conjunction. Subordinating
    markers (although/because/if) are already inside the advcl subtree and
    flow through naturally.
    """

    def __init__(self) -> None:
        try:
            import spacy

            self.nlp = spacy.load("en_core_web_sm")
        except (OSError, ImportError) as exc:
            print(f"spaCy unavailable ({exc}). Using regex fallback.")
            self.nlp = None

    def segment(self, text: str, source_input_id: str) -> List[Clause]:
        if self.nlp is None:
            return _fallback_segment(text, source_input_id)

        doc = self.nlp(text)
        clauses: List[Clause] = []
        for sent_idx, sent in enumerate(doc.sents):
            for clause in _clauses_from_sentence(
                sent, sent_idx, source_input_id, len(clauses)
            ):
                clauses.append(clause)
        return clauses


def _clauses_from_sentence(
    sent, sent_idx: int, source_input_id: str, position_offset: int
) -> List[Clause]:
    heads = _find_clause_heads(sent)
    if not heads:
        return []

    head_ids = {h.i for h in heads}

    # `cc` tokens (but/and/or) are children of the antecedent verb in spaCy's
    # parse, not the conjunct's. We transfer each one to its conjunct's clause
    # exclusively so it doesn't appear in both clauses.
    cc_transfers: dict[int, int] = {}
    for h in heads:
        if h.dep_ == "conj":
            cc = _connecting_cc(h)
            if cc is not None:
                cc_transfers[cc.i] = h.i

    out: List[Clause] = []

    for local_idx, head in enumerate(heads):
        tokens = _clause_tokens(head, head_ids)
        tokens = [
            t for t in tokens if cc_transfers.get(t.i, head.i) == head.i
        ]

        marker: Optional[str] = None
        if head.dep_ == "conj":
            cc_id = next(
                (k for k, v in cc_transfers.items() if v == head.i), None
            )
            if cc_id is not None:
                cc_tok = sent.doc[cc_id]
                if cc_id not in {t.i for t in tokens}:
                    tokens = sorted(tokens + [cc_tok], key=lambda t: t.i)
                marker = cc_tok.text
        elif head.dep_ == "advcl":
            marker = next(
                (c.text for c in head.children if c.dep_ == "mark"), None
            )
        elif head.dep_ == "relcl":
            marker = next(
                (
                    c.text
                    for c in head.subtree
                    if c.tag_ in _WH_TAGS and c.i < head.i
                ),
                None,
            )

        text = _detokenize(tokens)
        if not text:
            continue

        kind = _KIND_FROM_DEP.get(head.dep_, "fragment")

        out.append(
            Clause(
                id=f"{source_input_id}:{sent_idx}#{local_idx}",
                text=text,
                position=position_offset + local_idx,
                source_input_id=source_input_id,
                kind=kind,
                marker=marker,
                debug={
                    "head_index": head.i,
                    "head_text": head.text,
                    "dep": head.dep_,
                    "pos": head.pos_,
                },
            )
        )
    return out


def _find_clause_heads(sent) -> list:
    heads = []
    for tok in sent:
        if tok.dep_ == "ROOT":
            heads.append(tok)
        elif tok.dep_ in _CLAUSE_DEPS and tok.pos_ in _CLAUSE_HEAD_POS:
            heads.append(tok)
    heads.sort(key=lambda t: t.i)
    return heads


def _clause_tokens(head, head_ids) -> list:
    """Subtree of `head` minus subtrees rooted at other clause heads."""
    keep = {t.i: t for t in head.subtree}
    for other_id in head_ids:
        if other_id == head.i or other_id not in keep:
            continue
        for descendant in keep[other_id].subtree:
            keep.pop(descendant.i, None)
    return sorted(keep.values(), key=lambda t: t.i)


def _connecting_cc(conj_token):
    """The `cc` child of the antecedent that connects it to this conjunct."""
    parent = conj_token.head
    candidates = [
        c for c in parent.children if c.dep_ == "cc" and c.i < conj_token.i
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.i)


def _detokenize(tokens) -> str:
    """Reconstruct readable text from a token list. Trims separator-only
    punctuation at the edges; preserves spaCy's whitespace_ when tokens are
    adjacent in the source doc."""
    if not tokens:
        return ""

    start = 0
    while (
        start < len(tokens)
        and tokens[start].is_punct
        and tokens[start].text in _SEPARATOR_PUNCT
    ):
        start += 1
    end = len(tokens)
    while (
        end > start
        and tokens[end - 1].is_punct
        and tokens[end - 1].text in _SEPARATOR_PUNCT
    ):
        end -= 1
    trimmed = tokens[start:end]
    if not trimmed:
        return ""

    pieces: list[str] = []
    for i, tok in enumerate(trimmed):
        if i == len(trimmed) - 1:
            pieces.append(tok.text)
            continue
        nxt = trimmed[i + 1]
        if nxt.i == tok.i + 1:
            pieces.append(tok.text_with_ws)
        else:
            pieces.append(tok.text)
            if not nxt.is_punct or nxt.text not in _TIGHT_PUNCT:
                pieces.append(" ")
    return "".join(pieces).strip()


# --- Fallback regex segmenter, only used when spaCy can't be loaded ---

_FALLBACK_MARKERS = (
    r"\b(but|and|or|because|although|though|while|since|if|unless|"
    r"when|where|that|which)\b"
)


def _fallback_segment(text: str, source_input_id: str) -> List[Clause]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    units: list[str] = []
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        parts = re.split(_FALLBACK_MARKERS, s, flags=re.IGNORECASE)
        head = parts[0].strip() if parts else ""
        if head:
            units.append(head)
        for i in range(1, len(parts) - 1, 2):
            marker = parts[i]
            tail = parts[i + 1].strip()
            if tail:
                units.append(f"{marker} {tail}")
    return [
        Clause(
            id=f"{source_input_id}:{idx}",
            text=unit.rstrip(" .,;:"),
            position=idx,
            source_input_id=source_input_id,
            kind="fragment",
        )
        for idx, unit in enumerate(units)
        if unit.strip()
    ]


def create_simple_segmenter() -> SpacyClauseSegmenter:
    """Factory function to ensure consistent segmenter creation."""
    return SpacyClauseSegmenter()
