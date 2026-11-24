# diag_compare.py — Recherche de l'ORIGINAL vs recherche AMÉLIORÉE (sans aucune API)
#
#   ORIGINAL : lecteur PDF d'Agno (1 chunk par page), recherche par le sens, 10 résultats
#   AMÉLIORÉ : découpage par clause (smart_chunking.py), recherche hybride
#              (sens + mots-clés BM25), 5 résultats
# Mêmes embeddings locaux, Qdrant en mémoire, aucun LLM, aucune clé, aucun quota.
#
# Usage :
#   .venv\Scripts\python.exe diag_compare.py Orion-Kestrel-Services-Agreement.pdf orion
#   .venv\Scripts\python.exe diag_compare.py Lumen-Harbor-SaaS-Agreement.pdf lumen
#   .venv\Scripts\python.exe diag_compare.py Meridian-Polder-Fleet-Agreement.pdf meridian
import sys

from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.vectordb.qdrant import Qdrant
from agno.vectordb.search import SearchType

from smart_chunking import build_documents

# (question, texte qui doit se trouver dans le passage pertinent), par contrat de test
QUESTION_SETS = {
    "orion": [
        ("What is the notice period for early termination, and which party can use it?", "sixty (60) days"),
        ("How many days does a party have to cure a material breach?", "fifteen (15) days after receiving"),
        ("Is there a termination fee?", "No other termination fee"),
        ("What is the penalty for late payment?", "1.5%"),
        ("Which state's law governs this agreement?", "State of Colorado"),
        ("What is the liability cap?", "six (6) months"),
        ("How long does confidentiality last after termination?", "three (3) years"),
        ("Can a party hire the other party's employees?", "solicit"),
        ("What does section 10.1 say?", "10.1"),
        ("What does section 4.3 say?", "4.3"),
        ("Where is the amount of USD 8,000 mentioned?", "8,000"),
    ],
    "lumen": [
        ("What is the notice period for early termination, and which party can use it?", "ninety (90) days"),
        ("Can the Provider terminate for convenience?", "Provider may not terminate"),
        ("Is there an early termination fee?", "fifty percent (50%)"),
        ("What happens if an invoice is paid late?", "eight percent (8%)"),
        ("What availability does the Provider guarantee?", "99.5% of the time"),
        ("How fast must a data breach be notified?", "seventy-two (72) hours"),
        ("Does the liability cap always apply?", "This limit does not apply"),
        ("How are disputes resolved?", "binding arbitration"),
        ("What does section 7.2 say?", "7.2"),
        ("What does section 10.1 say?", "10.1"),
        ("Where is the amount of USD 120,000 mentioned?", "120,000"),
    ],
}
QUESTION_SETS["meridian"] = [
    ("What is the notice period for early termination, and which party can use it?", "one hundred and twenty (120) days"),
    ("Can the Supplier terminate for convenience?", "only after the end of the second contract year"),
    ("Is there a termination charge?", "three (3) months of the average monthly Charges"),
    ("How many days to remedy a material breach before termination?", "remedied within twenty (20) days"),
    ("How much notice is required for a price increase?", "sixty (60) days written notice of the price increase"),
    ("When can the Supplier suspend the Services?", "ten (10) days written notice of suspension"),
    ("What is the liability cap?", "one hundred and fifty percent (150%)"),
    ("Does the liability cap apply to confidentiality breaches?", "shall not apply to liability for death"),
    ("What is the monthly price per vehicle?", "EUR 38 per vehicle"),
    ("Which courts have jurisdiction?", "courts of Amsterdam"),
    ("What does section 21.2 say?", "21.2"),
    ("What does section 19.3 say?", "19.3"),
]
QUESTIONS = []

EMBEDDER = dict(id="BAAI/bge-small-en-v1.5", dimensions=384)


def evaluate(label, docs, search_type, limit):
    vdb = Qdrant(collection=label, location=":memory:", search_type=search_type,
                 embedder=FastEmbedEmbedder(**EMBEDDER))
    vdb.create()
    vdb.insert(content_hash=label, documents=docs)
    rows, top1, top3, chars, tested = [], 0, 0, 0, 0
    for q, expected in QUESTIONS:
        if not any(expected.lower() in d.content.lower() for d in docs):
            rows.append((q, "absent"))
            continue
        tested += 1
        results = vdb.search(q, limit=limit)
        ranks = [i for i, d in enumerate(results, 1) if expected.lower() in d.content.lower()]
        rank = ranks[0] if ranks else None
        top1 += rank == 1
        top3 += bool(rank and rank <= 3)
        chars += sum(len(d.content) for d in results)
        rows.append((q, rank))
    return {"label": label, "chunks": len(docs),
            "avg": sum(len(d.content) for d in docs) // max(len(docs), 1),
            "rows": rows, "top1": top1, "top3": top3, "tested": tested,
            "chars": chars // max(tested, 1)}


def fmt(rank):
    if rank == "absent":
        return "absent"
    return "1er ✅" if rank == 1 else f"{rank}e ⚠️" if rank and rank <= 3 else (
        f"{rank}e ❌" if rank else "non trouvé ❌")


def main(pdf_path, set_name):
    QUESTIONS[:] = QUESTION_SETS[set_name]
    original = evaluate("original", PDFReader().read(pdf_path), SearchType.vector, 10)
    improved = evaluate("ameliore", build_documents(pdf_path, pdf_path),
                        SearchType.hybrid, 5)

    print(f"\nDocument : {pdf_path}\n")
    print(f"{'':34}{'ORIGINAL':>22}{'AMÉLIORÉ':>22}")
    print(f"{'Chunks':34}{original['chunks']:>22}{improved['chunks']:>22}")
    print(f"{'Taille moyenne (caractères)':34}{original['avg']:>22}{improved['avg']:>22}")
    print("-" * 78)
    print(f"{'Position du passage attendu':34}")
    for (q, r1), (_, r2) in zip(original["rows"], improved["rows"]):
        print(f"  {q[:50]:<50}{fmt(r1):>12}{fmt(r2):>14}")
    print("-" * 78)
    n = original["tested"]
    print(f"{'Bon passage en 1re position':34}{original['top1']:>19}/{n}{improved['top1']:>19}/{n}")
    print(f"{'Bon passage dans le top 3':34}{original['top3']:>19}/{n}{improved['top3']:>19}/{n}")
    print(f"{'Texte envoyé au LLM / recherche':34}{original['chars']:>15} car.{improved['chars']:>17} car.")
    print(f"{'  soit environ':34}{original['chars'] // 4:>15} tok.{improved['chars'] // 4:>17} tok.")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[2] not in QUESTION_SETS:
        sys.exit("Usage : python diag_compare.py chemin/vers/contrat.pdf orion|lumen|meridian")
    main(sys.argv[1], sys.argv[2])