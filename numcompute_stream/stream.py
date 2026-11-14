"""Streaming orchestration and prequential evaluation module for NumCompute framework.
Handles chunk-by-chunk stream processing following the test-then-train paradigm.
"""

import numpy as np

__all__ = ["StreamEvaluator"]


class StreamEvaluator:
    """Evaluates and trains a pipeline or estimator incrementally using a prequential 
    (test-then-train) approach over a sequence of streaming data chunks.
    """

    def __init__(self, model, metric):
        """Initialize the evaluator with a streaming-compatible model/pipeline 
        and a streaming metric tracker.
        """
        self.model = model
        self.metric = metric
        self.history_ = []

    def evaluate_stream(self, data_stream):
        """Process a stream of chunks (X_chunk, y_chunk), predicting on incoming data 
        before incrementally updating the model and metric.
        ____________
        Parameters:
            data_stream : iterable of tuples (X_chunk, y_chunk) -> incoming data stream.
        ------------
        Returns :
            list -> History of metric scores across chunks.
        """
        self.history_ = []
        self.metric.reset()

        for X_chunk, y_chunk in data_stream:
            X_arr = np.asarray(X_chunk)
            y_arr = np.asarray(y_chunk)

            if X_arr.size == 0 or y_arr.size == 0:
                continue

            # Step 1: Predict first if the model is already fitted (Test phase)
            try:
                y_pred = self.model.predict(X_arr)
                self.metric.update(y_arr, y_pred)
                self.history_.append(self.metric.get())
            except Exception:
                pass

            # Step 2: Train/update incrementally afterwards (Train phase)
            if hasattr(self.model, "partial_fit"):
                self.model.partial_fit(X_arr, y_arr)
            elif hasattr(self.model, "fit"):
                self.model.fit(X_arr, y_arr)

        return self.history_