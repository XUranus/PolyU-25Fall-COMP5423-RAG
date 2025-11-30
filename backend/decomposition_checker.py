import re
import spacy

class DecompositionChecker:
    def __init__(self):
        # Load spaCy model once (lightweight model is fine)
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])  # faster

        # Common multi-hop lexical triggers
        self.multi_hop_keywords = {
            'relative_pronouns': {'that', 'which', 'who', 'whom', 'whose', 'where', 'when'},
            'conjunctions': {'and', 'or'},
            'prepositions': {'of', 'in', 'on', 'for', 'with', 'by', 'about', 'from'},
        }

        # Regex for common compositional patterns
        self.patterns = [
            r'\b(what|who|which|how|where|when)\b.*\b(that|which|who|whose|where|when)\b',
            r'\b\w+\s+(of|in|on|for|with|by|about|from)\s+.*\b(what|who|which|how|where|when)\b',
            r'\b(what|who|which|how|where|when).*\b(and|or)\b.*\b(what|who|which|how|where|when)\b',
        ]


    def identify_multi_hop_pattern(self, question: str) -> bool:
        question_lower = question.lower().strip()
        if not question_lower.endswith('?'):
            question_lower += '?'  # Normalize

        # 1. Regex-based pattern matching
        for pattern in self.patterns:
            if re.search(pattern, question_lower):
                return True

        # 2. Dependency-based clause detection (more robust)
        doc = self.nlp(question)
        num_clauses = sum(1 for token in doc if token.dep_ in ("relcl", "ccomp", "xcomp", "advcl"))
        if num_clauses >= 1:
            return True

        # 3. Heuristic: multiple prepositions or conjunctions + interrogative
        tokens = [token.text.lower() for token in doc]
        interrogatives = {"what", "who", "which", "how", "where", "when", "why"}
        has_interrogative = any(tok in interrogatives for tok in tokens)
        if not has_interrogative:
            return False

        rel_pronouns = self.multi_hop_keywords['relative_pronouns']
        if any(tok in rel_pronouns for tok in tokens):
            return True

        prep_count = sum(1 for tok in tokens if tok in self.multi_hop_keywords['prepositions'])
        conj_count = sum(1 for tok in tokens if tok in self.multi_hop_keywords['conjunctions'])

        # Simple heuristic: multiple prepositions often signal composition
        if prep_count >= 2 or conj_count >= 2:
            return True

        return False