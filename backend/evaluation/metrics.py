import math
import statistics


class InferenceMetrics:
    def __init__(self):
        self.started = None
        self.chunks = []
        self.usage = {}

    def observe(self, event):
        if event["type"] == "inference_started":
            self.started = event["at"]
        elif event["type"] == "visible_chunk":
            self.chunks.append(event["at"])
        elif event["type"] == "usage":
            self.usage = {key: value for key, value in event.items() if key not in ("type", "at")}

    def result(self):
        count, duration = self.usage.get("eval_count"), self.usage.get("eval_duration")
        seconds = duration / 1e9 if duration is not None else None
        intervals = [right - left for left, right in zip(self.chunks, self.chunks[1:])]
        return dict(inference_ttft_ms=(self.chunks[0] - self.started) * 1000 if self.chunks and self.started is not None else None,
                    tps=count / seconds if count and seconds else None,
                    tpot_ms=seconds * 1000 / count if count and seconds is not None else None,
                    provider_output_tokens=count, provider_usage=self.usage,
                    visible_chunks=len(self.chunks), true_token_itl_ms=None,
                    stream_chunk_interval_ms=statistics.mean(intervals) * 1000 if intervals else None)


METRICS = ("ttft_ms", "inference_ttft_ms", "tps", "tpot_ms", "e2e_seconds", "planner_ms", "retrieval_ms", "reranker_ms")


def summarize(records):
    result = {}
    for name in METRICS:
        values = sorted(item[name] for item in records if item.get("status") == "completed"
                        and isinstance(item.get(name), (int, float)) and math.isfinite(item[name]))
        result[name] = dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                            p95=values[max(0, math.ceil(.95 * len(values)) - 1)],
                            variance=statistics.variance(values) if len(values) > 1 else None) if values else None
    return result
