"""
Node Health package (Person B) for FAP-Scale.

Contains:
- failure_scorer: CPI, TTS, LightGBM-based node failure probability scorer
- interference: 15-bin co-location interference matrix with EMA updates
"""

from .failure_scorer import NodeFailureScorer, score_node_failure, calculate_cpi, calculate_tts
from .interference import InterferenceMatrix, get_interference

__all__ = [
    "NodeFailureScorer",
    "score_node_failure",
    "calculate_cpi",
    "calculate_tts",
    "InterferenceMatrix",
    "get_interference",
]
