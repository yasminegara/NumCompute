"""Visualization module for the NumCompute framework.
Provides plotting utilities for streaming evaluation metrics and model comparisons.
"""

import matplotlib.pyplot as plt

__all__ = ["plot_streaming_metrics"]


def plot_streaming_metrics(evaluation_results):
    """Plot prequential evaluation metrics over time for different buffer sizes.
    
    Parameters:
        evaluation_results : dict -> Mapping of buffer sizes to lists of metric scores.
    """
    plt.figure(figsize=(10, 6))
    
    for buffer_size, history in evaluation_results.items():
        plt.plot(history, marker="o", label=f"Buffer Size: {buffer_size}")
        
    plt.xlabel("Chunk Index")
    plt.ylabel("Streaming Metric Score")
    plt.title("Prequential Evaluation Performance Across Chunks")
    plt.legend()
    plt.grid(True)
    plt.show()