"""
smart_chunking.py — Découpage par clause qui garde le numéro de page (sans API).

Pourquoi : le lecteur PDF d'Agno fait 1 chunk par page. Une page mélange plusieurs
clauses, une clause peut être coupée entre deux pages, et l'agent reçoit des pages
entières (beaucoup de tokens).

Ici :
  1. on recolle tout le texte du contrat en retenant où commence chaque page ;
  2. on coupe à chaque début de clause ("6.1. Notwithstanding", "ARTICLE V", ...) ;
  3. chaque chunk garde sa page de départ ET le titre de sa section parente,
     ex. "[6. TERMINATION] 6.1.1. In the event ..." ;
  4. une clause trop longue est redécoupée avec chevauchement.
Sur un texte sans numérotation, on retombe sur des morceaux de taille fixe.
"""
import hashlib
import re
from bisect import bisect_right
from typing import List

from agno.knowledge.document.base import Document
from pypdf import PdfReader

# "6. TERMINATION", "6.1. Notwithstanding", "ARTICLE V", "SECTION 12.1"
CLAUSE_HEADER = re.compile(
    # pas un renvoi interne ("under Article 9.", "see Section 4.")
    r"(?<!Article )(?<!article )(?<!Section )(?<!section )(?<!ARTICLE )(?<!SECTION )"
    r"(?:(?<=\s)|^)(?:"
    r"(?:ARTICLE|SECTION)\s+[0-9IVXLC]+(?:\.[0-9]+)*\.?\s+(?=[A-Z])"
    r"|(?:ATTACHMENT|EXHIBIT|SCHEDULE|APPENDIX)\s+[A-Z0-9]{1,3}\b"
    r"|[0-9]{1,2}(?:\.[0-9]{1,2}){0,3}\.\s+(?=[A-Z(])"
    r")"
)
# Titre de section de premier niveau : "6. TERMINATION", "ARTICLE V"
TOP_LEVEL = re.compile(r"^(?:[0-9]{1,2}\.\s+[A-Z][A-Z ,&/-]{3,}(?=\s[0-9A-Z][a-z0-9.])|"
                       r"(?:ARTICLE|SECTION)\s+[0-9IVXLC]+\b|"
                       r"(?:ATTACHMENT|EXHIBIT|SCHEDULE|APPENDIX)\s+[A-Z0-9]{1,3}\b)")
SECTION_NUMBER = re.compile(r"(?:^|\s)((?:ARTICLE|SECTION|ATTACHMENT|EXHIBIT|SCHEDULE|APPENDIX)\s+[0-9A-Z.]{1,6}"
                            r"|[0-9]{1,2}(?:\.[0-9]{1,2}){1,3}|[0-9]{1,2}(?=\.\s))")
PAGE_MARKER = re.compile(r"<(?:start|end) page \d+>")

MAX_SIZE = 1200   # caractères max par chunk
OVERLAP = 150     # chevauchement quand une clause longue est redécoupée
MIN_SIZE = 60     # un titre seul est rattaché à la clause suivante


def _split_long(text: str) -> List[str]:
    if len(text) <= MAX_SIZE:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(start + MAX_SIZE, len(text))
        cut = text.rfind(". ", start + MAX_SIZE // 2, end)
        if end < len(text) and cut != -1:
            end = cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - OVERLAP, start + 1)
    return pieces


def build_documents(pdf_path: str, name: str) -> List[Document]:
    # Lecture directe avec pypdf : le lecteur d'Agno supprime le nombre en début de
    # page (il le prend pour un numéro de page) et peut ainsi effacer "1." d'un article.
    pages = [p.extract_text() or "" for p in PdfReader(pdf_path).pages]

    # 1. texte complet + position de début de chaque page
    full, starts, page_numbers = "", [], []
    for i, text in enumerate(pages, 1):
        starts.append(len(full))
        page_numbers.append(i)
        full += " ".join(PAGE_MARKER.sub(" ", text).split()) + " "

    def page_at(pos: int) -> int:
        return page_numbers[max(bisect_right(starts, pos) - 1, 0)]

    # 2. découpage aux débuts de clause
    cuts = [m.start() for m in CLAUSE_HEADER.finditer(full)]
    if not cuts or cuts[0] != 0:
        cuts = [0] + cuts
    segments = [(a, full[a:b].strip()) for a, b in zip(cuts, cuts[1:] + [len(full)])]

    # on rattache les segments trop courts (titre seul) au suivant
    merged, buf_pos, buf = [], None, ""
    for pos, seg in segments:
        if not seg:
            continue
        if buf:
            buf, pos = f"{buf} {seg}", buf_pos
        else:
            buf = seg
        if len(buf) >= MIN_SIZE:
            merged.append((pos, buf))
            buf = ""
        else:
            buf_pos = pos
    if buf:
        merged.append((buf_pos or 0, buf))

    # 3. chunks avec page, numéro de section et titre parent
    docs, parent = [], ""
    for pos, clause in merged:
        top = TOP_LEVEL.match(clause)
        if top:
            parent = top.group(0).strip()
        # numéro de la clause : le dernier titre trouvé au début
        # (ex. "6. TERMINATION 6.1. Notwithstanding" -> 6.1 ; "ARTICLE 4. LEVELS 4.1." -> 4.1)
        heads = [m.group(0).strip().rstrip(".").strip() for m in CLAUSE_HEADER.finditer(clause[:160])]
        section = heads[-1] if heads else ""
        for i, piece in enumerate(_split_long(clause)):
            prefix = f"[{parent}] " if parent and not piece.startswith(parent) else ""
            content = f"{prefix}{piece}"
            meta = {"page": page_at(pos), "section": section, "chunk": len(docs) + 1}
            docs.append(Document(
                id=hashlib.md5(f"{name}|{len(docs)}|{content[:50]}".encode()).hexdigest(),
                name=name,
                meta_data=meta,
                content=content,
            ))
    return docs