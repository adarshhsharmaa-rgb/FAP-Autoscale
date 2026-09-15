"""
Data generation package for FAP-Scale simulation.
"""

from .synthetic_data import generate_workload_time_series, generate_node_telemetry, extract_single_node_telemetry

__all__ = ["generate_workload_time_series", "generate_node_telemetry", "extract_single_node_telemetry"]
