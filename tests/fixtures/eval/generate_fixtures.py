"""Generate committed eval fixture PNGs (solid-color proxies for CI)."""

from pathlib import Path

from PIL import Image

FIXTURES = Path(__file__).resolve().parent


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (384, 384), color=(180, 140, 90)).save(
        FIXTURES / "animal_warm.png"
    )
    Image.new("RGB", (384, 384), color=(40, 120, 200)).save(FIXTURES / "scene_cool.png")


if __name__ == "__main__":
    main()
