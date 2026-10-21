def top_n(scores: dict[str, float], n: int = 3) -> list[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]]
