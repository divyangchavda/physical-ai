"""CLIP compatibility shim for Ultralytics YOLO-World.

Ultralytics expects the original OpenAI CLIP API (clip.load()).
We only have open_clip installed. This module provides a compatibility layer.
"""
import open_clip
import torch


def load(name, device="cpu", download_root=None):
    """Mimic clip.load() using open_clip.

    Args:
        name: Model name (e.g., "ViT-B/32")
        device: Device to load model on
        download_root: Ignored (open_clip downloads to its own cache)

    Returns:
        (model, preprocess) tuple matching clip.load() API
    """
    # Map CLIP names to open_clip equivalents
    name_map = {
        "ViT-B/32": ("ViT-B-32", "openai"),
        "ViT-B/16": ("ViT-B-16", "openai"),
        "ViT-L/14": ("ViT-L-14", "openai"),
    }

    if name in name_map:
        model_name, pretrained = name_map[name]
    else:
        # Fallback: try to parse
        model_name = name.replace("/", "-")
        pretrained = "openai"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )

    return model, preprocess


def tokenize(texts, truncate=True):
    """Mimic clip.tokenize() using open_clip.
    
    Args:
        texts: List of strings or single string to tokenize
        truncate: Ignored (open_clip always handles truncation)
    """
    return open_clip.tokenize(texts)


# Expose open_clip functions that Ultralytics might need
available_models = open_clip.list_pretrained
