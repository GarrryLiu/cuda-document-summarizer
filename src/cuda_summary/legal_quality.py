"""Source-grounding triage, NOT legal entailment or factuality scoring.

A duration found elsewhere in the source can still be assigned to the wrong
party/clause in a summary. Absence of a flag never means a claim is supported.
"""
import re
import difflib

WORDS = dict(zip('zero one two three four five six seven eight nine ten eleven twelve'.split(), range(13)))
WORDS.update(dict(zip('thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(),range(13,20))))
for tens,value in zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(),range(20,100,10)):
    WORDS[tens]=value
    for unit in list(WORDS):
        if 1<=WORDS[unit]<=9:
            WORDS[tens+' '+unit]=value+WORDS[unit]
            WORDS[tens+'-'+unit]=value+WORDS[unit]
NUMBER = r'(?:\d+|' + '|'.join(sorted(WORDS,key=len,reverse=True)) + r')'
DURATION = re.compile(r'\b(' + NUMBER + r')(?:\s*\(\s*(\d+)\s*\))?[\s-]+(?:(business|calendar)\s+)?(day|week|month|year|hour)s?\b', re.I)
LEGAL_MARKERS=re.compile(r'\b(?:not|no|shall|must|may|except\w*|unless|surviv\w*|law\w*|court\w*|terminat\w*|days?|months?|years?)\b|subject to',re.I)


def sensitive_changes(reference, candidate):
    """Changed lines containing legal markers; a review queue, not error detection."""
    before=reference.splitlines();after=candidate.splitlines();result=[]
    for op,i,j,k,l in difflib.SequenceMatcher(None,before,after).get_opcodes():
        if op!='equal' and LEGAL_MARKERS.search('\n'.join(before[i:j]+after[k:l])):
            result.append({'reference_lines':before[i:j],'candidate_lines':after[k:l]})
    return result


def durations(text):
    result=[]
    for m in DURATION.finditer(text):
        number=m[2] or m[1].lower()
        value=WORDS[number] if number in WORDS else int(number)
        result.append((value,(m[3] or '').lower(),m[4].lower()))
    ordinals={'first':1,'second':2,'third':3,'fourth':4,'fifth':5}
    for ordinal in re.findall(r'\b(first|second|third|fourth|fifth)\s+anniversary\b',text,re.I):
        result.append((ordinals[ordinal.lower()],'','year'))
    return set(result)


def triage(source, summary):
    """Return review flags, with no pass/fail quality judgment."""
    source_durations=durations(source)
    missing=sorted(durations(summary)-source_durations)
    # Numeric anchors are only lexical: dates/entities/subjects are not aligned.
    def numbers(text):
        text=re.sub(r'\b(\d+)(?:st|nd|rd|th)\b',r'\1',text,flags=re.I)
        return set(re.findall(r'\b\d+(?:\.\d+)?\b',text))
    novel_numbers=sorted(numbers(summary)-numbers(source))
    return {'duration_mismatch_count':len(missing),
            'numeric_anchor_mismatch_count':len(novel_numbers),
            'over_160_words':len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b",summary))>160,
            'review_required':bool(missing or novel_numbers),
            'unmatched_durations':missing,'unmatched_numeric_anchors':novel_numbers}
