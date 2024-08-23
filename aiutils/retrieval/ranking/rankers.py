from rank_bm25 import BM25Okapi
from typing import List, Dict
import json

def reciprocal_rank_fusion(documents: List[Dict[str, str]],
                           rankings: List[Dict[str, int]],
                           id_key: str = "id",
                           k: int = 60,
                           precision: int = 5) -> List[Dict[str, str]]:
  """A simple implementation of Reciprocal Rank Fusion.

  For details, check out the paper:
  https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
  """

  def score_document(document: Dict[str, str]) -> float:
    # Aggregates score for a document across all given rankings
    nonlocal rankings
    doc_id = document[id_key]
    return sum([1.0 / (k + ranking[doc_id])
                for ranking in rankings
                if doc_id in ranking])
  # Calculate the scores for all documents
  scores = list(map(score_document, documents))
  # Add the RRF score to the documents and sort by highest score
  fused_docs = sorted([{**doc, "rrf_score": round(score, precision)}
                        for doc, score in zip(documents, scores)],
                      key=lambda doc: doc["rrf_score"], reverse=True)
  return fused_docs