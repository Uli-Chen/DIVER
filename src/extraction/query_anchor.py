"""Narrow, auditable quote-escape normalization of a fixed observed anchor.

Never decode arbitrary Unicode escapes, remove punctuation, repair graph labels,
substitute aliases, or rewrite unrelated parts of the generated question.
"""
import re

PROTOCOL = 'anchor-single-quote-escape-v1'


def canonicalize_anchor(query, anchor, observed_labels=()):
    if not isinstance(anchor, str) or not anchor:
        raise ValueError('A nonempty fixed anchor is required')
    # An optional single backslash is accepted only immediately before a quote
    # that is literally present in the known anchor. All other bytes are fixed.
    pieces = [(r'(?:\\)?' + re.escape(c)) if c in ('"', "'") else re.escape(c) for c in anchor]
    pattern = re.compile(r'(?<!\w)' + ''.join(pieces) + r'(?!\w)', re.IGNORECASE)
    matches = list(pattern.finditer(query))
    if not matches:
        raise ValueError('Exploit query writer omitted or changed the fixed anchor')
    labels = {str(label).casefold() for label in observed_labels}
    for match in matches:
        text = match.group(0)
        if text.casefold() != anchor.casefold() and text.casefold() in labels:
            raise ValueError('Escaped anchor is ambiguous with another observed label')
    canonical = pattern.sub(lambda match: anchor, query)
    return canonical, {'protocol': PROTOCOL, 'matched_spellings': [m.group(0) for m in matches],
                       'changed': canonical != query}
