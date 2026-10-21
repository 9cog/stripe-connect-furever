from thumbgen import target_size


def test_landscape_scales_to_longest_edge():
    assert target_size(1000, 500, 256) == (256, 128)


def test_portrait_scales_to_longest_edge():
    assert target_size(300, 600, 100) == (50, 100)
