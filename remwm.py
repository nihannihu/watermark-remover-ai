import sys
import os
import json

# Print immediately so the GUI knows we're alive
print("Initializing...", flush=True)

import click
from pathlib import Path

print("Loading image libraries...", flush=True)
import cv2
import numpy as np
from PIL import Image, ImageDraw

print("Loading AI framework (PyTorch)...", flush=True)
import torch
from torch.nn import Module

print("Loading AI models (Transformers)...", flush=True)
# Monkey-patch: cached_download was removed in huggingface_hub 0.24, add compatibility shim
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

from transformers import AutoProcessor, AutoModelForCausalLM

print("Loading inpainting engine...", flush=True)
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config

import tqdm
from loguru import logger
from enum import Enum
import tempfile
import shutil
import subprocess
import uuid

print("All modules loaded.", flush=True)

try:
    from cv2.typing import MatLike
except ImportError:
    MatLike = np.ndarray


def get_ffmpeg_cmd():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"

def download_lama_model():
    """Download LaMA model using iopaint."""
    logger.info("Downloading LaMA model... (this may take a few minutes)")
    print("Downloading LaMA model (~196MB)... Please wait.")

    result = subprocess.run(
        [sys.executable, "-m", "iopaint", "download", "--model", "lama"],
        capture_output=False,  # Show download progress
        text=True
    )

    if result.returncode != 0:
        logger.error("Failed to download LaMA model")
        return False

    logger.info("LaMA model downloaded successfully")
    print("LaMA model downloaded!")
    return True


def load_lama_model(device):
    """Load LaMA model, downloading if necessary."""
    try:
        return ModelManager(name="lama", device=device)
    except NotImplementedError as e:
        if "Unsupported model: lama" in str(e):
            print("LaMA model not available, attempting to download...")
            if download_lama_model():
                # Re-import to refresh model registry
                import importlib
                import iopaint.model
                importlib.reload(iopaint.model)
                # Try again
                return ModelManager(name="lama", device=device)
            else:
                raise RuntimeError("Failed to download LaMA model. Please run manually: python\\python.exe -m iopaint download --model lama")
        raise

class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    OCR_WITH_REGION = "<OCR_WITH_REGION>"
    """Detect bounding box for objects and OCR text"""

def identify(task_prompt: TaskType, image: MatLike, text_input: str, model: AutoModelForCausalLM, processor: AutoProcessor, device: str):
    if not isinstance(task_prompt, TaskType):
        raise ValueError(f"task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)}")

    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    # Cast tensors to match the model's dtype and move to the correct device
    inputs = {k: v.to(device).to(model.dtype) if v.is_floating_point() else v.to(device)
              for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=1,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.width, image.height)
    )

def extract_ocr_bboxes(image: MatLike, model: AutoModelForCausalLM, processor: AutoProcessor, device: str, max_bbox_percent: float):
    """
    Detect individual text regions using Florence-2 OCR_WITH_REGION.
    Returns oriented quad polygons to prevent rectangular over-masking on angled watermarks.
    Supports multi-crop scanning for high-resolution images so faint/corner watermarks are never missed.
    """
    w, h = image.size
    image_area = w * h
    ocr_key = "<OCR_WITH_REGION>"

    all_quads = []
    all_labels = []

    # 1. Full image OCR
    try:
        parsed = identify(TaskType.OCR_WITH_REGION, image, None, model, processor, device)
        if ocr_key in parsed:
            data = parsed[ocr_key]
            for q, l in zip(data.get("quad_boxes", []), data.get("labels", [])):
                all_quads.append(q)
                all_labels.append(l)
    except Exception as e:
        logger.warning(f"Full image OCR error: {e}")

    # 2. Quadrant crops for images to catch corner and small watermarks
    if w >= 600 or h >= 600:
        crops = [
            (0, 0, int(w * 0.65), int(h * 0.65)),
            (int(w * 0.35), 0, w, int(h * 0.65)),
            (0, int(h * 0.35), int(w * 0.65), h),
            (int(w * 0.35), int(h * 0.35), w, h),
        ]
        for x_off, y_off, x2, y2 in crops:
            try:
                crop_img = image.crop((x_off, y_off, x2, y2))
                res_crop = identify(TaskType.OCR_WITH_REGION, crop_img, None, model, processor, device)
                if ocr_key in res_crop:
                    d_crop = res_crop[ocr_key]
                    for q, l in zip(d_crop.get("quad_boxes", []), d_crop.get("labels", [])):
                        if len(q) >= 8:
                            global_q = [
                                q[0] + x_off, q[1] + y_off,
                                q[2] + x_off, q[3] + y_off,
                                q[4] + x_off, q[5] + y_off,
                                q[6] + x_off, q[7] + y_off
                            ]
                            all_quads.append(global_q)
                            all_labels.append(l)
            except Exception:
                pass

    bboxes = []
    seen_centers = []

    for lbl, q in zip(all_labels, all_quads):
        if len(q) >= 8:
            xs = [q[0], q[2], q[4], q[6]]
            ys = [q[1], q[3], q[5], q[7]]
            cx = sum(xs) / 4.0
            cy = sum(ys) / 4.0

            # Deduplicate quads with centers within 25px
            is_dup = False
            for prev_cx, prev_cy in seen_centers:
                if (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2 < 25 ** 2:
                    is_dup = True
                    break
            if is_dup:
                continue
            seen_centers.append((cx, cy))

            p0 = np.array([q[0], q[1]], dtype=float)
            p1 = np.array([q[2], q[3]], dtype=float)
            p2 = np.array([q[4], q[5]], dtype=float)
            p3 = np.array([q[6], q[7]], dtype=float)

            v_top = p1 - p0
            len_top = np.linalg.norm(v_top)
            u_top = v_top / len_top if len_top > 0 else np.array([1.0, 0.0])

            v_bot = p2 - p3
            len_bot = np.linalg.norm(v_bot)
            u_bot = v_bot / len_bot if len_bot > 0 else np.array([1.0, 0.0])

            clean_lbl = lbl.replace("</s>", "").strip().upper()

            # Extend partial watermark words along orientation vector
            is_prefix = any(p in clean_lbl for p in ["WATER", "WATE", "WAT", "SHUTTER", "GETTY", "ISTOCK", "ADOBE", "ALAMY", "PREVIEW"]) and not any(s in clean_lbl for s in ["MARK", "STOCK", "IMAGES"])
            is_suffix = any(s in clean_lbl for s in ["MARK", "ERMARK", "RMARK", "ARK", "STOCK"]) and not any(p in clean_lbl for p in ["WATER", "WATE", "SHUTTER"])
            is_middle = any(m == clean_lbl for m in ["TER", "ERM", "ERMA"])

            if is_prefix:
                p1 += u_top * min(len_top * 1.2, 220.0)
                p2 += u_bot * min(len_bot * 1.2, 220.0)
            elif is_suffix:
                p0 -= u_top * min(len_top * 1.2, 220.0)
                p3 -= u_bot * min(len_bot * 1.2, 220.0)
            elif is_middle:
                p0 -= u_top * min(len_top * 1.0, 160.0)
                p3 -= u_bot * min(len_bot * 1.0, 160.0)
                p1 += u_top * min(len_top * 1.0, 160.0)
                p2 += u_bot * min(len_bot * 1.0, 160.0)

            # Clamp points inside image boundaries
            poly_pts = [
                max(0, min(w, int(p0[0]))), max(0, min(h, int(p0[1]))),
                max(0, min(w, int(p1[0]))), max(0, min(h, int(p1[1]))),
                max(0, min(w, int(p2[0]))), max(0, min(h, int(p2[1]))),
                max(0, min(w, int(p3[0]))), max(0, min(h, int(p3[1])))
            ]

            all_x = [poly_pts[0], poly_pts[2], poly_pts[4], poly_pts[6]]
            all_y = [poly_pts[1], poly_pts[3], poly_pts[5], poly_pts[7]]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            # Shoelace formula for polygon area
            poly_area = 0.5 * abs(
                poly_pts[0]*poly_pts[3] + poly_pts[2]*poly_pts[5] + poly_pts[4]*poly_pts[7] + poly_pts[6]*poly_pts[1]
                - (poly_pts[1]*poly_pts[2] + poly_pts[3]*poly_pts[4] + poly_pts[5]*poly_pts[6] + poly_pts[7]*poly_pts[0])
            )
            area_percent = (poly_area / image_area) * 100.0
            accepted = area_percent <= max_bbox_percent

            bboxes.append({
                "bbox": [min_x, min_y, max_x, max_y],
                "polygon": poly_pts,
                "area_percent": round(area_percent, 2),
                "accepted": accepted,
                "label": clean_lbl
            })

    return bboxes

def detect_only(image: MatLike, model: AutoModelForCausalLM, processor: AutoProcessor, device: str, max_bbox_percent: float, detection_prompt: str = "watermark"):
    """
    Detect watermarks and return bounding boxes/polygons WITHOUT creating mask or inpainting.
    Used for preview mode and mask creation.
    Combines open vocabulary detection with high-precision oriented OCR polygons.
    """
    # 1. Always run OCR extraction for precise oriented quad polygons
    ocr_results = extract_ocr_bboxes(image, model, processor, device, max_bbox_percent)

    # 2. Run open vocabulary detection (for logos, icons, graphical watermarks)
    vocab_results = []
    try:
        task_prompt = TaskType.OPEN_VOCAB_DETECTION
        parsed_answer = identify(task_prompt, image, detection_prompt, model, processor, device)
        detection_key = "<OPEN_VOCABULARY_DETECTION>"

        if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
            image_area = image.width * image.height
            for bbox in parsed_answer[detection_key]["bboxes"]:
                x1, y1, x2, y2 = map(int, bbox)
                bbox_area = (x2 - x1) * (y2 - y1)
                area_percent = (bbox_area / image_area) * 100
                accepted = area_percent <= max_bbox_percent

                # CHECK FOR OVERLAP WITH OCR:
                # If this vocab bbox overlaps with any OCR text polygon, DISCARD IT!
                # Why? An axis-aligned box over angled text encompasses surrounding faces/background!
                overlaps_ocr = False
                for ocr_det in ocr_results:
                    ox1, oy1, ox2, oy2 = ocr_det["bbox"]
                    ix1 = max(x1, ox1)
                    iy1 = max(y1, oy1)
                    ix2 = min(x2, ox2)
                    iy2 = min(y2, oy2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        if inter_area > 0.25 * min(bbox_area, (ox2 - ox1) * (oy2 - oy1)):
                            overlaps_ocr = True
                            break

                if not overlaps_ocr:
                    vocab_results.append({
                        "bbox": [x1, y1, x2, y2],
                        "area_percent": round(area_percent, 2),
                        "accepted": accepted
                    })
    except Exception as e:
        logger.warning(f"Vocab detection error: {e}")

    # Combine: OCR polygons take precedence, non-overlapping logo boxes added
    return ocr_results + vocab_results

def get_watermark_mask(image: MatLike, model: AutoModelForCausalLM, processor: AutoProcessor, device: str, max_bbox_percent: float, detection_prompt: str = "watermark"):
    """
    Detect watermarks and create a mask for inpainting.
    Uses stroke-level extraction over face/skin regions to strictly protect lips, nose, and facial features.
    Uses solid tight polygons over non-facial regions to ensure 100% complete text removal.
    """
    w, h = image.size
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    detections = detect_only(image, model, processor, device, max_bbox_percent, detection_prompt)

    # Detect faces to protect facial anatomy (lips, philtrum, nose, eyes)
    face_zone = np.zeros((h, w), dtype=np.uint8)
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray_full, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        for (fx, fy, fw, fh) in detected_faces:
            x1 = max(0, int(fx - 0.4 * fw))
            y1 = max(0, int(fy - 0.15 * fh))
            x2 = min(w, int(fx + 1.6 * fw))
            y2 = min(h, int(fy + 2.0 * fh))
            cv2.rectangle(face_zone, (x1, y1), (x2, y2), 255, -1)
    except Exception as e:
        logger.warning(f"Face cascade detection skipped: {e}")

    # Check for repeating lattice watermarks (multiple angled text instances)
    # If detected, propagate the lattice grid across the image so corner/faint watermarks are never missed
    angled_quads = []
    for det in detections:
        if det.get("accepted") and "polygon" in det and len(det["polygon"]) >= 8:
            angled_quads.append(det["polygon"])

    if len(angled_quads) >= 2:
        try:
            # Estimate dominant angle, filtering out quads cut off by image borders
            angles = []
            for q in angled_quads:
                touches_border = any(q[i] <= 2 or q[i] >= w-2 for i in range(0, 8, 2)) or any(q[i] <= 2 or q[i] >= h-2 for i in range(1, 8, 2))
                dx = q[2] - q[0]
                dy = q[3] - q[1]
                if abs(dx) > 15 and not touches_border:
                    angles.append(np.degrees(np.arctan2(dy, dx)))
            if not angles:
                for q in angled_quads:
                    dx = q[2] - q[0]
                    dy = q[3] - q[1]
                    if abs(dx) > 15:
                        angles.append(np.degrees(np.arctan2(dy, dx)))

            if angles:
                median_angle = float(np.median(angles))
                # Rotate and detect grid spacing
                M_rot = cv2.getRotationMatrix2D((w/2, h/2), median_angle, 1.0)
                cos_a = np.abs(M_rot[0, 0])
                sin_a = np.abs(M_rot[0, 1])
                nW = int((h * sin_a) + (w * cos_a))
                nH = int((h * cos_a) + (w * sin_a))
                M_rot[0, 2] += (nW / 2) - w/2
                M_rot[1, 2] += (nH / 2) - h/2
                M_inv = cv2.invertAffineTransform(M_rot)

                # Project quads to rotated frame
                centers_rot = []
                sizes = []
                for q in angled_quads:
                    pts = np.array([(q[i], q[i+1]) for i in range(0, 8, 2)], dtype=np.float32)
                    pts_rot = cv2.transform(np.array([pts]), M_rot)[0]
                    xs = pts_rot[:, 0]
                    ys = pts_rot[:, 1]
                    centers_rot.append((np.min(xs), np.min(ys)))
                    sizes.append((np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)))

                tw = int(np.median([s[0] for s in sizes])) if sizes else 380
                th = int(np.median([s[1] for s in sizes])) if sizes else 60
                tw = max(tw, 445)  # encompass brackets/chevrons
                th = max(th, 68)

                # If repeating pattern detected, add lattice grid points
                if len(centers_rot) >= 2:
                    v1_candidates = []
                    v2_candidates = []
                    for i in range(len(centers_rot)):
                        for j in range(len(centers_rot)):
                            if i != j:
                                c_dx = centers_rot[j][0] - centers_rot[i][0]
                                c_dy = centers_rot[j][1] - centers_rot[i][1]
                                if 250 < c_dx < 500 and 80 < c_dy < 200:
                                    v1_candidates.append((c_dx, c_dy))
                                if 10 < c_dx < 120 and 220 < c_dy < 400:
                                    v2_candidates.append((c_dx, c_dy))

                    v1 = (float(np.median([c[0] for c in v1_candidates])), float(np.median([c[1] for c in v1_candidates]))) if v1_candidates else (374.0, 139.0)
                    v2 = (float(np.median([c[0] for c in v2_candidates])), float(np.median([c[1] for c in v2_candidates]))) if v2_candidates else (45.0, 300.0)

                    base_x, base_y = centers_rot[0]
                    for m in range(-3, 4):
                        for n in range(-3, 4):
                            gx = base_x + m * v1[0] + n * v2[0]
                            gy = base_y + m * v1[1] + n * v2[1]
                            corners_r = np.array([
                                [gx, gy], [gx + tw, gy],
                                [gx + tw, gy + th], [gx, gy + th]
                            ], dtype=np.float32)
                            corners_o = cv2.transform(np.array([corners_r]), M_inv)[0]
                            xs_o = corners_o[:, 0]
                            ys_o = corners_o[:, 1]
                            if xs_o.max() >= 0 and xs_o.min() < w and ys_o.max() >= 0 and ys_o.min() < h:
                                poly_o = [
                                    int(corners_o[0, 0]), int(corners_o[0, 1]),
                                    int(corners_o[1, 0]), int(corners_o[1, 1]),
                                    int(corners_o[2, 0]), int(corners_o[2, 1]),
                                    int(corners_o[3, 0]), int(corners_o[3, 1])
                                ]
                                detections.append({
                                    "bbox": [int(xs_o.min()), int(ys_o.min()), int(xs_o.max()), int(ys_o.max())],
                                    "polygon": poly_o,
                                    "accepted": True
                                })
        except Exception as e:
            logger.warning(f"Lattice expansion skipped: {e}")

    # Build smart mask
    for det in detections:
        if not det.get("accepted"):
            continue

        poly = det.get("polygon")
        if not poly or len(poly) < 8:
            if "bbox" in det:
                x1, y1, x2, y2 = det["bbox"]
                poly = [x1, y1, x2, y1, x2, y2, x1, y2]
            else:
                continue

        pts = np.array([(poly[i], poly[i+1]) for i in range(0, 8, 2)], dtype=np.float32)
        tw = int(np.linalg.norm(pts[1] - pts[0]))
        th = int(np.linalg.norm(pts[3] - pts[0]))

        if tw < 5 or th < 5:
            continue

        poly_box = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_box, [np.int32(pts)], 255)
        overlap_area = np.sum((poly_box > 0) & (face_zone > 0))
        poly_area = max(1, np.sum(poly_box > 0))
        is_face_watermark = (overlap_area / float(poly_area)) > 0.05

        if is_face_watermark:
            # STROKE-LEVEL MASK OVER FACE/LIPS - NO TOPHAT!
            # Strictly isolate watermark text strokes without damaging lips, nostril rims, or philtrum highlights
            try:
                M_strip = cv2.getAffineTransform(pts[:3], np.float32([[0, 0], [tw, 0], [tw, th]]))
                strip = cv2.warpAffine(img_bgr, M_strip, (tw, th))
                strip_face = cv2.warpAffine(face_zone, M_strip, (tw, th))
                gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

                thresh = ((gray_strip > 210) | (gray_strip < 40)).astype(np.uint8) * 255
                nb_comp, output, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
                clean = np.zeros_like(thresh)
                for i in range(1, nb_comp):
                    if stats[i, cv2.CC_STAT_AREA] > 10:
                        clean[output == i] = 255
                kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                dil_letters = cv2.dilate(clean, kernel_dil)

                strip_final = np.zeros((th, tw), dtype=np.uint8)
                strip_final[strip_face == 0] = 255
                strip_final[(strip_face > 0) & (dil_letters > 0)] = 255

                M_inv_strip = cv2.getAffineTransform(np.float32([[0, 0], [tw, 0], [tw, th]]), pts[:3])
                box_mask = cv2.warpAffine(strip_final, M_inv_strip, (w, h), flags=cv2.INTER_NEAREST)
                full_mask = cv2.bitwise_or(full_mask, box_mask)
            except Exception:
                full_mask = cv2.bitwise_or(full_mask, poly_box)
        else:
            # SOLID POLYGON OVER BACKGROUND, CLOTHING, STREET SIGNS, VEHICLES
            kernel_box = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            solid_box = cv2.dilate(poly_box, kernel_box)
            full_mask = cv2.bitwise_or(full_mask, solid_box)

    return Image.fromarray(full_mask)

def process_image_with_lama(image: MatLike, mask: MatLike, model_manager: ModelManager):
    h, w = image.shape[:2]
    strategy = HDStrategy.ORIGINAL if max(h, w) <= 1600 else HDStrategy.CROP
    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=strategy,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    result = model_manager(image, mask, config)

    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def make_region_transparent(image: Image.Image, mask: Image.Image):
    image = image.convert("RGBA")
    mask = mask.convert("L")
    transparent_image = Image.new("RGBA", image.size)
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) > 0:
                transparent_image.putpixel((x, y), (0, 0, 0, 0))
            else:
                transparent_image.putpixel((x, y), image.getpixel((x, y)))
    return transparent_image

def is_video_file(file_path):
    """Check if the file is a video based on its extension"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return Path(file_path).suffix.lower() in video_extensions

def process_video(input_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt="watermark", progress_offset=0, progress_scale=100):
    """Process a video file by extracting frames, removing watermarks, and reconstructing the video"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine output format
    if force_format:
        output_format = force_format.upper()
    else:
        output_format = "MP4"  # Default to MP4 for videos
    
    # Create output video file
    output_path = Path(output_path)
    if output_path.is_dir():
        output_file = output_path / f"{input_path.stem}_no_watermark.{output_format.lower()}"
    else:
        output_file = output_path.with_suffix(f".{output_format.lower()}")
    
    # Create a temporary file for the video without audio
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"
    
    # Set codec based on output format
    if output_format.upper() == "MP4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_format.upper() == "AVI":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Default to MP4
    
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
    
    # Process each frame
    with tqdm.tqdm(total=total_frames, desc="Processing video frames") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Get watermark mask
            mask_image = get_watermark_mask(pil_image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt)
            
            # Process frame
            if transparent:
                # For video, we can't use transparency, so we'll fill with a color or background
                result_image = make_region_transparent(pil_image, mask_image)
                # Convert RGBA to RGB by filling transparent areas with white
                background = Image.new("RGB", result_image.size, (255, 255, 255))
                background.paste(result_image, mask=result_image.split()[3])
                result_image = background
            else:
                lama_result = process_image_with_lama(np.array(pil_image), np.array(mask_image), model_manager)
                result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))
            
            # Convert back to OpenCV format and write to output video
            frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            out.write(frame_result)
            
            # Update progress
            frame_count += 1
            pbar.update(1)
            local_progress = frame_count / total_frames
            progress = int(progress_offset + local_progress * progress_scale)
            print(f"Processing frame {frame_count}/{total_frames} ({int(local_progress*100)}%), overall_progress:{progress}%", flush=True)
    
    # Release resources
    cap.release()
    out.release()
    
    # Combine processed video with original audio using FFmpeg
    try:
        logger.info("Merging processed video with original audio...")
        
        # Check if FFmpeg is available
        try:
            subprocess.check_output([get_ffmpeg_cmd(), "-version"], stderr=subprocess.STDOUT)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg is not available. Video will be produced without audio.")
            shutil.copy(str(temp_video_path), str(output_file))
        else:
            # Use FFmpeg to combine processed video with original audio
            ffmpeg_cmd = [
                get_ffmpeg_cmd(), "-y",
                "-i", str(temp_video_path),  # Processed video without audio
                "-i", str(input_path),       # Original video with audio
                "-c:v", "copy",              # Copy video without re-encoding
                "-c:a", "aac",               # Encode audio as AAC for better compatibility
                "-map", "0:v:0",             # Use video track from first file (processed video)
                "-map", "1:a:0",             # Use audio track from second file (original video)
                "-shortest",                  # End when the shortest track ends
                str(output_file)
            ]

            # Execute FFmpeg
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Audio/video merge completed successfully!")
    except Exception as e:
        logger.error(f"Error during audio/video merge: {str(e)}")
        # In case of error, use video without audio
        shutil.copy(str(temp_video_path), str(output_file))
    finally:
        # Clean up temporary files
        try:
            os.remove(str(temp_video_path))
            os.rmdir(temp_dir)
        except:
            pass
    
    final_progress = progress_offset + progress_scale
    logger.info(f"input_path:{input_path}, output_path:{output_file}, overall_progress:{final_progress}")
    return output_file


def process_video_two_pass(input_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt="watermark", detection_skip=1, fade_in_sec=0.0, fade_out_sec=0.0, progress_offset=0, progress_scale=100):
    """
    Two-pass video processing with frame skip detection and fade in/out handling.

    Pass 1: Detect watermarks every N frames (sparse detection)
    Pass 2: Apply inpainting to all frames using interpolated masks

    This is more efficient for videos where watermarks don't change rapidly,
    and handles fade in/out watermarks by extending the mask temporally.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert seconds to frames
    fade_in_frames = int(fade_in_sec * fps)
    fade_out_frames = int(fade_out_sec * fps)

    logger.info(f"Two-pass processing: {total_frames} frames, skip={detection_skip}, fade_in={fade_in_frames}f, fade_out={fade_out_frames}f")

    # Determine output format
    if force_format:
        output_format = force_format.upper()
    else:
        output_format = "MP4"

    # Create output video file
    output_path = Path(output_path)
    if output_path.is_dir():
        output_file = output_path / f"{input_path.stem}_no_watermark.{output_format.lower()}"
    else:
        output_file = output_path.with_suffix(f".{output_format.lower()}")

    # ========== PASS 1: DETECTION (sparse) ==========
    logger.info("Pass 1: Detecting watermarks...")
    detections = {}  # frame_idx -> [bbox, bbox, ...]
    detection_frames = list(range(0, total_frames, detection_skip))

    with tqdm.tqdm(total=len(detection_frames), desc="Pass 1: Detection") as pbar:
        for frame_idx in detection_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            bboxes = detect_only(pil_image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt)

            if bboxes:
                accepted_bboxes = [b["bbox"] for b in bboxes if b["accepted"]]
                if accepted_bboxes:
                    detections[frame_idx] = accepted_bboxes

            pbar.update(1)
            local_progress = (pbar.n / len(detection_frames)) * 0.5  # Pass 1 = 0-50% local
            progress = int(progress_offset + local_progress * progress_scale)
            print(f"Pass 1 (Detecting): frame {frame_idx}/{total_frames} ({int(local_progress*200)}%), overall_progress:{progress}%", flush=True)

    logger.info(f"Pass 1 complete: found watermarks in {len(detections)} detection points")

    # ========== TIMELINE EXPANSION ==========
    # Create frame->bbox mapping with fade in/out expansion
    frame_masks = {}  # frame_idx -> [bbox, ...]

    for det_frame, bboxes in detections.items():
        # Expand backwards (fade in) - watermark might be fading in before detection
        start_frame = max(0, det_frame - fade_in_frames)
        # Expand forwards (fade out) - continue masking after detection
        # Also include frames until next detection point
        end_frame = min(total_frames, det_frame + detection_skip + fade_out_frames)

        for f in range(start_frame, end_frame):
            if f not in frame_masks:
                frame_masks[f] = []
            # Add bboxes, avoiding duplicates
            for bbox in bboxes:
                if bbox not in frame_masks[f]:
                    frame_masks[f].append(bbox)

    logger.info(f"Timeline expanded: {len(frame_masks)} frames will have inpainting applied")

    # ========== PASS 2: INPAINTING ==========
    logger.info("Pass 2: Applying inpainting...")

    # Create temporary file for video without audio
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"

    # Set codec
    if output_format.upper() == "MP4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_format.upper() == "AVI":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    with tqdm.tqdm(total=total_frames, desc="Pass 2: Inpainting") as pbar:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_masks:
                # This frame needs inpainting
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Create mask from bboxes
                mask = Image.new("L", pil_image.size, 0)
                draw = ImageDraw.Draw(mask)
                for bbox in frame_masks[frame_idx]:
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([x1, y1, x2, y2], fill=255)

                # Apply inpainting or transparency
                if transparent:
                    result_image = make_region_transparent(pil_image, mask)
                    background = Image.new("RGB", result_image.size, (255, 255, 255))
                    background.paste(result_image, mask=result_image.split()[3])
                    result_image = background
                else:
                    lama_result = process_image_with_lama(np.array(pil_image), np.array(mask), model_manager)
                    result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

                frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            else:
                # No watermark detected for this frame, copy original
                frame_result = frame

            out.write(frame_result)
            frame_idx += 1
            pbar.update(1)
            local_progress = 0.5 + (frame_idx / total_frames) * 0.5  # Pass 2 = 50-100% local
            progress = int(progress_offset + local_progress * progress_scale)
            print(f"Pass 2 (Inpainting): frame {frame_idx}/{total_frames} ({int((local_progress-0.5)*200)}%), overall_progress:{progress}%", flush=True)

    cap.release()
    out.release()

    # ========== MERGE WITH AUDIO ==========
    try:
        logger.info("Merging processed video with original audio...")
        try:
            subprocess.check_output([get_ffmpeg_cmd(), "-version"], stderr=subprocess.STDOUT)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg is not available. Video will be produced without audio.")
            shutil.copy(str(temp_video_path), str(output_file))
        else:
            ffmpeg_cmd = [
                get_ffmpeg_cmd(), "-y",
                "-i", str(temp_video_path),
                "-i", str(input_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                str(output_file)
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Audio/video merge completed successfully!")
    except Exception as e:
        logger.error(f"Error during audio/video merge: {str(e)}")
        shutil.copy(str(temp_video_path), str(output_file))
    finally:
        try:
            os.remove(str(temp_video_path))
            os.rmdir(temp_dir)
        except:
            pass

    final_progress = progress_offset + progress_scale
    logger.info(f"input_path:{input_path}, output_path:{output_file}, overall_progress:{final_progress}")
    return output_file


def handle_one(image_path: Path, output_path: Path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite, detection_prompt="watermark", detection_skip=1, fade_in=0.0, fade_out=0.0, progress_offset=0, progress_scale=100):
    # SAFETY: Never overwrite the input file
    if image_path.resolve() == output_path.resolve():
        logger.error(f"Cannot overwrite input file: {image_path}. Choose a different output path.")
        print(f"ERROR: Cannot overwrite input file! Choose a different output folder.")
        return

    if output_path.exists() and not overwrite:
        logger.info(f"Skipping existing file: {output_path}")
        return

    # Check if it's a video file
    if is_video_file(image_path):
        # Use two-pass if detection_skip > 1 or fade handling is needed
        use_two_pass = detection_skip > 1 or fade_in > 0 or fade_out > 0
        if use_two_pass:
            return process_video_two_pass(image_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt, detection_skip, fade_in, fade_out, progress_offset, progress_scale)
        else:
            return process_video(image_path, output_path, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, detection_prompt, progress_offset, progress_scale)

    # Process image
    image = Image.open(image_path).convert("RGB")
    mask_image = get_watermark_mask(image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt)

    if transparent:
        result_image = make_region_transparent(image, mask_image)
    else:
        lama_result = process_image_with_lama(np.array(image), np.array(mask_image), model_manager)
        result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

    # Determine output format
    if force_format:
        output_format = force_format.upper()
    elif transparent:
        output_format = "PNG"
    else:
        output_format = image_path.suffix[1:].upper()
        if output_format not in ["PNG", "WEBP", "JPG"]:
            output_format = "PNG"
    
    # Map JPG to JPEG for PIL compatibility
    if output_format == "JPG":
        output_format = "JPEG"

    if transparent and output_format == "JPG":
        logger.warning("Transparency detected. Defaulting to PNG for transparency support.")
        output_format = "PNG"

    new_output_path = output_path.with_suffix(f".{output_format.lower()}")
    result_image.save(new_output_path, format=output_format)
    # Report progress for this image (end of range)
    final_progress = progress_offset + progress_scale
    print(f"input_path:{image_path}, output_path:{new_output_path}, overall_progress:{final_progress}%")
    return new_output_path

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(), required=False, default=None)
@click.option("--preview", is_flag=True, help="Preview mode: detect watermarks and output JSON with base64 image (no processing).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files in bulk mode.")
@click.option("--transparent", is_flag=True, help="Make watermark regions transparent instead of removing.")
@click.option("--max-bbox-percent", default=10.0, help="Maximum percentage of the image that a bounding box can cover.")
@click.option("--force-format", type=click.Choice(["PNG", "WEBP", "JPG", "MP4", "AVI"], case_sensitive=False), default=None, help="Force output format. Defaults to input format.")
@click.option("--detection-prompt", default="watermark", help="Text prompt for watermark detection (e.g. 'watermark', 'watermark Sora logo', 'Getty Images').")
@click.option("--detection-skip", default=1, type=int, help="Detect watermarks every N frames for videos (1-10). Higher = faster but may miss brief watermarks.")
@click.option("--fade-in", default=0.0, type=float, help="Extend mask backwards by N seconds to handle fade-in watermarks.")
@click.option("--fade-out", default=0.0, type=float, help="Extend mask forwards by N seconds to handle fade-out watermarks.")
def main(input_path: str, output_path: str, preview: bool, overwrite: bool, transparent: bool, max_bbox_percent: float, force_format: str, detection_prompt: str, detection_skip: int, fade_in: float, fade_out: float):
    # Input validation
    if detection_skip < 1 or detection_skip > 10:
        logger.warning(f"detection_skip must be 1-10, got {detection_skip}. Using 1.")
        detection_skip = max(1, min(10, detection_skip))
    if fade_in < 0:
        fade_in = 0
    if fade_out < 0:
        fade_out = 0

    input_path = Path(input_path)

    # ========== PREVIEW MODE ==========
    if preview:
        import json
        import base64
        from io import BytesIO
        import random

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Force no dtype for CUDA (intentional default)
        # Apply float32 for CPU (compatibility)
        model_dtype = torch.float32 if device == "cpu" else None

        florence_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True,
            torch_dtype=model_dtype).to(device).eval()
        florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

        # Get sample image from input
        if input_path.is_dir():
            # Get a random image from directory
            images = list(input_path.glob("*.[jp][pn]g")) + list(input_path.glob("*.webp"))
            videos = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi")) + list(input_path.glob("*.mov"))
            files = images + videos
            if not files:
                print(json.dumps({"error": "No supported files found in directory"}))
                return
            sample_path = random.choice(files)
        else:
            sample_path = input_path

        # Load image (extract frame if video)
        if is_video_file(sample_path):
            cap = cv2.VideoCapture(str(sample_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Get frame from middle of video
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(json.dumps({"error": f"Could not read frame from video: {sample_path}"}))
                return
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            source_type = "video"
            source_frame = total_frames // 2
        else:
            pil_image = Image.open(sample_path).convert("RGB")
            source_type = "image"
            source_frame = None

        # Run detection
        detections = detect_only(pil_image, florence_model, florence_processor, device, max_bbox_percent, detection_prompt)

        # Draw bounding boxes on image
        draw = ImageDraw.Draw(pil_image)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 255, 0) if det["accepted"] else (255, 0, 0)  # Green if accepted, red if rejected
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Draw label
            label = f"{det['area_percent']:.1f}%"
            draw.text((x1, y1 - 15), label, fill=color)

        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Output JSON result
        result = {
            "image": img_base64,  # Just base64, GUI adds prefix
            "detections": detections,
            "source": str(sample_path),
            "source_type": source_type,
            "source_frame": source_frame,
            "prompt_used": detection_prompt,
            "max_bbox_percent": max_bbox_percent
        }
        print(json.dumps(result))
        return

    # ========== NORMAL PROCESSING MODE ==========
    output_path = Path(output_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # Force no dtype for CUDA (intentional default)
    # Apply float32 for CPU (compatibility)
    model_dtype = torch.float32 if device == "cpu" else None

    print("Loading Florence-2 model... (first run downloads ~1.5GB)", flush=True)
    florence_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True,
        torch_dtype=model_dtype).to(device).eval()
    florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    print("Florence-2 model loaded.", flush=True)
    logger.info("Florence-2 Model loaded")

    if not transparent:
        print("Loading LaMa inpainting model...", flush=True)
        model_manager = load_lama_model(device)
        print("LaMa model loaded.", flush=True)
        logger.info("LaMa model loaded")
    else:
        model_manager = None

    if input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True)

        # Include video files in the search
        images = list(input_path.glob("*.[jp][pn]g")) + list(input_path.glob("*.webp"))
        videos = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi")) + list(input_path.glob("*.mov")) + list(input_path.glob("*.mkv"))
        files = images + videos
        total_files = len(files)

        for idx, file_path in enumerate(tqdm.tqdm(files, desc="Processing files")):
            output_file = output_path / file_path.name
            # Scale the batch progress from 65% to 100% since 0-65% is used for model loading
            progress_offset = 65 + int(idx / total_files * 35)
            progress_scale = int(35 / total_files)
            handle_one(file_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite, detection_prompt, detection_skip, fade_in, fade_out, progress_offset, progress_scale)
    else:
        # Single file mode - if output is a directory, construct file path
        if output_path.is_dir():
            output_file = output_path / input_path.name
        else:
            output_file = output_path

        # Ensure video output has proper extension
        if is_video_file(input_path) and output_file.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
            if force_format and force_format.upper() in ["MP4", "AVI"]:
                output_file = output_file.with_suffix(f".{force_format.lower()}")
            else:
                output_file = output_file.with_suffix(".mp4")  # Default to mp4

        # Start at 65% since 0-65% was used for model loading
        handle_one(input_path, output_file, florence_model, florence_processor, model_manager, device, transparent, max_bbox_percent, force_format, overwrite, detection_prompt, detection_skip, fade_in, fade_out, progress_offset=65, progress_scale=35)
        print(f"input_path:{input_path}, output_path:{output_file}, overall_progress:100")

if __name__ == "__main__":
    main()
