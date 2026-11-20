"""Generate ``packaging/photoseder.ico`` (a simple lens icon) with Pillow."""

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
GREEN = (47, 111, 94, 255)
WHITE = (255, 255, 255, 255)


def draw_icon() -> Image.Image:
    """Return the icon artwork: a white lens on a rounded green square."""
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, SIZE - 8, SIZE - 8), radius=52, fill=GREEN)
    centre = SIZE // 2
    for radius, width in ((84, 14), (50, 12)):
        box = (centre - radius, centre - radius, centre + radius, centre + radius)
        draw.ellipse(box, outline=WHITE, width=width)
    draw.ellipse((centre - 18, centre - 18, centre + 18, centre + 18), fill=WHITE)
    return image


def main() -> None:
    """Write the multi-size ``.ico`` next to this script."""
    target = Path(__file__).with_name("photoseder.ico")
    sizes = [(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]
    draw_icon().save(target, format="ICO", sizes=sizes)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
