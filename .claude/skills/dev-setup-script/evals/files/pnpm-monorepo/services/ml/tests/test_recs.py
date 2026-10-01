from recs import top_n


def test_top_n_orders_by_score():
    assert top_n({"a": 0.1, "b": 0.9, "c": 0.5}, 2) == ["b", "c"]
