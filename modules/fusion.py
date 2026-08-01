from config import FusionConfig

class FusionModule:
    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg

    def fuse(self, scene_type: str, sparse_count: float, dense_count: float) -> float:
        primary, secondary = (
            (sparse_count, dense_count) if scene_type == "sparse" else (dense_count, sparse_count)
        )
        primary_w, secondary_w = (
            (self.cfg.sparse_weight, self.cfg.dense_weight)
            if scene_type == "sparse"
            else (self.cfg.dense_weight, self.cfg.sparse_weight)
        )

        if primary <= 0:
            return max(primary, secondary)

        disagreement = abs(primary - secondary) / primary
        if disagreement > self.cfg.disagreement_fallback:
            # Branches disagree too much to blend meaningfully; trust the
            # branch the density-decision module chose as authoritative.
            return primary

        total_w = primary_w + secondary_w
        return (primary * primary_w + secondary * secondary_w) / total_w
