from PIL import Image


def target_size(width: int, height: int, longest: int) -> tuple[int, int]:
    scale = longest / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def render(path: str, longest: int) -> str:
    img = Image.open(path)
    img.thumbnail(target_size(*img.size, longest))
    out = path.rsplit(".", 1)[0] + f".{longest}.png"
    img.save(out)
    return out
