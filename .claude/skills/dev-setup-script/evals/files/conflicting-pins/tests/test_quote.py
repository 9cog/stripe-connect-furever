from pricing import quote


def test_no_discount():
    assert quote(250, 4) == 1000


def test_discount_rounds_half_up():
    assert quote(333, 3, 10) == 899
