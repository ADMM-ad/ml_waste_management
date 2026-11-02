"""
prepare_dataset.py
Curates and generates a standardized 300-image dataset (100 Organic, 100 Plastic, 100 Paper)
split into train (70/70/70), val (15/15/15), and test (15/15/15) with high domain diversity.
"""
import os
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGE_SIZE = (224, 224)

CATEGORIES = ["organic", "plastic", "paper"]
SPLITS = {
    "train": 70,
    "val": 15,
    "test": 15
}

def create_organic_image(seed_val: int) -> Image.Image:
    """Generates realistic visual features for Organic Waste (leaves, fruits, peels, organic decay)."""
    rng = np.random.RandomState(seed_val)
    bg_color = (
        int(rng.randint(60, 110)),
        int(rng.randint(50, 95)),
        int(rng.randint(35, 75))
    )
    img = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(img)

    element_types = ["leaf", "fruit_blob", "peel_strip", "decay_spot"]
    num_elements = int(rng.randint(4, 9))
    for _ in range(num_elements):
        el_type = element_types[int(rng.randint(0, len(element_types)))]
        if el_type == "leaf":
            color = (int(rng.randint(30, 90)), int(rng.randint(110, 200)), int(rng.randint(20, 70)))
            center = (int(rng.randint(30, 190)), int(rng.randint(30, 190)))
            rx, ry = int(rng.randint(15, 45)), int(rng.randint(8, 25))
            bbox = [center[0] - rx, center[1] - ry, center[0] + rx, center[1] + ry]
            draw.ellipse(bbox, fill=color, outline=(color[0]-15, color[1]-20, color[2]-10))
        elif el_type == "fruit_blob":
            fruit_palettes = [
                (int(rng.randint(190, 245)), int(rng.randint(90, 140)), int(rng.randint(20, 40))),
                (int(rng.randint(160, 220)), int(rng.randint(30, 70)), int(rng.randint(30, 60))),
                (int(rng.randint(200, 245)), int(rng.randint(180, 225)), int(rng.randint(30, 60))),
            ]
            color = fruit_palettes[int(rng.randint(0, len(fruit_palettes)))]
            center = (int(rng.randint(40, 180)), int(rng.randint(40, 180)))
            radius = int(rng.randint(15, 38))
            draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius], fill=color)
        elif el_type == "peel_strip":
            points = []
            cur_x, cur_y = int(rng.randint(20, 180)), int(rng.randint(20, 180))
            for _ in range(5):
                points.append((cur_x, cur_y))
                cur_x += int(rng.randint(-30, 30))
                cur_y += int(rng.randint(-30, 30))
            if len(points) >= 2:
                draw.line(points, fill=(int(rng.randint(170, 230)), int(rng.randint(130, 190)), 30), width=int(rng.randint(4, 9)))
        else:
            spot_color = (int(rng.randint(30, 60)), int(rng.randint(20, 45)), int(rng.randint(15, 30)))
            cx, cy = int(rng.randint(20, 200)), int(rng.randint(20, 200))
            rad = int(rng.randint(5, 18))
            draw.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], fill=spot_color)

    arr = np.array(img, dtype=np.int16)
    noise = rng.randint(-18, 18, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    res_img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.5))
    return res_img

def create_plastic_image(seed_val: int) -> Image.Image:
    """Generates visual features for Plastic Waste (bottles, cups, bags, shiny reflections, crinkles)."""
    rng = np.random.RandomState(seed_val)
    bg_val = int(rng.randint(130, 180))
    img = Image.new("RGB", IMAGE_SIZE, (bg_val, bg_val, bg_val + int(rng.randint(-5, 15))))
    draw = ImageDraw.Draw(img)

    plastic_types = ["bottle", "crinkled_bag", "cup_rings", "colored_container"]
    plastic_type = plastic_types[int(rng.randint(0, len(plastic_types)))]

    if plastic_type == "bottle":
        bottle_color = (int(rng.randint(140, 210)), int(rng.randint(200, 245)), int(rng.randint(230, 255)))
        bx, by = int(rng.randint(60, 120)), int(rng.randint(40, 90))
        bw, bh = int(rng.randint(45, 75)), int(rng.randint(90, 130))
        draw.rounded_rectangle([bx, by, bx+bw, by+bh], radius=10, fill=bottle_color, outline=(90, 140, 200), width=3)
        draw.rectangle([bx+bw//4, by-20, bx+3*bw//4, by], fill=(int(rng.randint(10, 50)), int(rng.randint(70, 200)), int(rng.randint(180, 240))))
        draw.line([(bx+8, by+10), (bx+8, by+bh-10)], fill=(255, 255, 255), width=int(rng.randint(3, 5)))
    elif plastic_type == "crinkled_bag":
        bag_palettes = [
            (230, 235, 240),
            (220, 60, 60),
            (40, 120, 220),
            (240, 220, 30)
        ]
        bag_base = bag_palettes[int(rng.randint(0, len(bag_palettes)))]
        poly_points = [
            (int(rng.randint(30, 70)), int(rng.randint(40, 80))),
            (int(rng.randint(140, 190)), int(rng.randint(30, 70))),
            (int(rng.randint(150, 200)), int(rng.randint(140, 190))),
            (int(rng.randint(40, 90)), int(rng.randint(150, 195)))
        ]
        draw.polygon(poly_points, fill=bag_base, outline=(bag_base[0]//2, bag_base[1]//2, bag_base[2]//2))
        for _ in range(8):
            p1 = (int(rng.randint(40, 180)), int(rng.randint(40, 180)))
            p2 = (p1[0] + int(rng.randint(-35, 35)), p1[1] + int(rng.randint(-35, 35)))
            draw.line([p1, p2], fill=(255, 255, 255), width=2)
    elif plastic_type == "cup_rings":
        cup_color = (int(rng.randint(200, 230)), int(rng.randint(220, 245)), int(rng.randint(235, 255)))
        cx, cy = int(rng.randint(80, 140)), int(rng.randint(80, 140))
        for r in range(int(rng.randint(45, 65)), 10, -8):
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(100, 160, 230), fill=cup_color, width=2)
            draw.arc([cx-r, cy-r, cx+r, cy+r], start=45, end=135, fill=(255, 255, 255), width=3)
    else:
        color = (int(rng.randint(20, 60)), int(rng.randint(120, 210)), int(rng.randint(160, 230)))
        rx1, ry1 = int(rng.randint(30, 70)), int(rng.randint(40, 80))
        rx2, ry2 = rx1 + int(rng.randint(70, 120)), ry1 + int(rng.randint(70, 110))
        draw.rectangle([rx1, ry1, rx2, ry2], fill=color, outline=(255, 255, 255), width=3)
        draw.rectangle([rx1+15, ry1+15, rx1+45, ry1+35], fill=(245, 245, 245))

    arr = np.array(img, dtype=np.uint8)
    return Image.fromarray(arr)

def create_paper_image(seed_val: int) -> Image.Image:
    """Generates visual features for Paper Waste (cardboard, newspaper, paper sheets, text rows, fibers)."""
    rng = np.random.RandomState(seed_val)
    paper_types = ["cardboard", "newspaper", "notebook_sheet", "folded_kraft"]
    paper_type = paper_types[int(rng.randint(0, len(paper_types)))]

    if paper_type in ["cardboard", "folded_kraft"]:
        base_color = (int(rng.randint(180, 215)), int(rng.randint(145, 180)), int(rng.randint(100, 135)))
        img = Image.new("RGB", IMAGE_SIZE, (100, 100, 100))
        draw = ImageDraw.Draw(img)
        cx, cy = int(rng.randint(20, 50)), int(rng.randint(20, 50))
        cw, ch = int(rng.randint(120, 160)), int(rng.randint(120, 160))
        draw.rectangle([cx, cy, cx+cw, cy+ch], fill=base_color, outline=(base_color[0]-40, base_color[1]-40, base_color[2]-30), width=3)
        for y_line in range(cy+8, cy+ch-8, int(rng.randint(8, 14))):
            draw.line([(cx+6, y_line), (cx+cw-6, y_line)], fill=(base_color[0]-25, base_color[1]-25, base_color[2]-20), width=2)
    else:
        paper_bg = (int(rng.randint(235, 248)), int(rng.randint(230, 245)), int(rng.randint(215, 235)))
        img = Image.new("RGB", IMAGE_SIZE, (90, 85, 80))
        draw = ImageDraw.Draw(img)
        sheet_poly = [
            (int(rng.randint(25, 55)), int(rng.randint(20, 50))),
            (int(rng.randint(165, 200)), int(rng.randint(25, 55))),
            (int(rng.randint(155, 195)), int(rng.randint(165, 200))),
            (int(rng.randint(20, 50)), int(rng.randint(155, 190)))
        ]
        draw.polygon(sheet_poly, fill=paper_bg, outline=(170, 170, 160), width=2)
        for y_pos in range(50, 165, int(rng.randint(6, 11))):
            draw.line([(45, y_pos), (int(rng.randint(140, 175)), y_pos)], fill=(45, 45, 45), width=int(rng.randint(1, 2)))
        draw.rectangle([45, 38, 130, 46], fill=(30, 30, 30))

    arr = np.array(img, dtype=np.int16)
    noise = rng.randint(-10, 10, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def main():
    print("Preparing 300-image dataset (Organic, Plastic, Paper)...")
    seed_counter = 1000

    for split_name, count in SPLITS.items():
        for category in CATEGORIES:
            cat_dir = DATASET_DIR / split_name / category
            cat_dir.mkdir(parents=True, exist_ok=True)
            print(f"Generating {count} images for [{split_name}/{category}]...")

            for i in range(1, count + 1):
                seed_counter += 1
                if category == "organic":
                    img = create_organic_image(seed_counter)
                elif category == "plastic":
                    img = create_plastic_image(seed_counter)
                else:
                    img = create_paper_image(seed_counter)

                file_path = cat_dir / f"{category}_{split_name}_{i:03d}.jpg"
                img.save(file_path, "JPEG", quality=92)

    print("Dataset generation completed successfully.")

if __name__ == "__main__":
    main()
