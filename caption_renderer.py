"""
Caption Renderer Module
Renders styled captions on video frames using word-level timestamps.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Caption style presets
CAPTION_STYLES = {
    "classic": {
        "font_scale": 1.2,
        "color": (255, 255, 255),  # White
        "outline_color": (0, 0, 0),  # Black
        "outline_thickness": 3,
        "background": None,
        "font_weight": "normal",
        "uppercase": False
    },
    "boxed": {
        "font_scale": 1.0,
        "color": (255, 255, 255),  # White
        "outline_color": None,
        "outline_thickness": 0,
        "background": (0, 0, 0, 180),  # Semi-transparent black
        "font_weight": "normal",
        "uppercase": False
    },
    "yellow": {
        "font_scale": 1.2,
        "color": (0, 255, 255),  # Yellow (BGR)
        "outline_color": (0, 0, 0),  # Black
        "outline_thickness": 3,
        "background": None,
        "font_weight": "normal",
        "uppercase": False
    },
    "minimal": {
        "font_scale": 0.9,
        "color": (255, 255, 255),  # White
        "outline_color": None,
        "outline_thickness": 0,
        "background": None,
        "font_weight": "light",
        "uppercase": False,
        "lowercase": True
    },
    "bold": {
        "font_scale": 1.5,
        "color": (255, 255, 255),  # White
        "outline_color": (0, 0, 0),  # Black
        "outline_thickness": 5,
        "background": None,
        "font_weight": "bold",
        "uppercase": True
    },
    "karaoke": {
        "font_scale": 1.2,
        "color": (255, 255, 255),  # White (inactive words)
        "highlight_color": (0, 255, 255),  # Yellow (active word)
        "outline_color": (0, 0, 0),
        "outline_thickness": 2,
        "background": None,
        "font_weight": "normal",
        "uppercase": False
    },
    "neon": {
        "font_scale": 1.2,
        "color": (255, 0, 255),  # Magenta/Pink
        "outline_color": (255, 100, 255),  # Lighter pink for glow
        "outline_thickness": 4,
        "glow": True,
        "background": None,
        "font_weight": "normal",
        "uppercase": False
    },
    "gradient": {
        "font_scale": 1.3,
        "color": (255, 255, 255),  # Base white
        "gradient_colors": [(255, 100, 100), (100, 100, 255)],  # Red to Blue
        "outline_color": (0, 0, 0),
        "outline_thickness": 2,
        "background": None,
        "font_weight": "normal",
        "uppercase": False
    }
}


def hex_to_bgr(hex_color):
    """Convert hex color string to BGR tuple for OpenCV."""
    if not hex_color:
        return None
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # BGR


def get_style_config(style_name, custom_color=None, custom_outline_color=None):
    """Get style configuration with optional custom colors."""
    if style_name not in CAPTION_STYLES:
        style_name = "classic"

    config = CAPTION_STYLES[style_name].copy()

    # Apply custom colors if provided
    if custom_color:
        config["color"] = hex_to_bgr(custom_color)
    if custom_outline_color:
        config["outline_color"] = hex_to_bgr(custom_outline_color)

    return config


def get_active_caption_text(transcript_words, current_time, window_size=3.0):
    """
    Get the words that should be displayed at the current timestamp.
    Returns a list of word dicts with timing info.
    """
    if not transcript_words:
        return []

    # Find words within the display window
    active_words = []
    for word in transcript_words:
        word_start = word.get('start', word.get('s', 0))
        word_end = word.get('end', word.get('e', 0))

        # Show word from its start until window_size after it ends
        if word_start <= current_time <= word_end + window_size:
            active_words.append({
                'word': word.get('word', word.get('w', '')),
                'start': word_start,
                'end': word_end,
                'is_current': word_start <= current_time <= word_end
            })

    return active_words


def wrap_text(text, max_width, font, font_scale, thickness):
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def render_text_with_outline(frame, text, position, font, font_scale, color, outline_color, outline_thickness):
    """Render text with outline effect."""
    x, y = position

    # Draw outline by rendering text multiple times offset
    if outline_color and outline_thickness > 0:
        for dx in range(-outline_thickness, outline_thickness + 1):
            for dy in range(-outline_thickness, outline_thickness + 1):
                if dx != 0 or dy != 0:
                    cv2.putText(frame, text, (x + dx, y + dy), font, font_scale,
                               outline_color, outline_thickness, cv2.LINE_AA)

    # Draw main text
    cv2.putText(frame, text, (x, y), font, font_scale, color, 2, cv2.LINE_AA)


def render_karaoke_text(frame, words, position, font, font_scale, config):
    """Render karaoke-style text with highlighted current word."""
    x, y = position
    current_x = x

    for i, word_info in enumerate(words):
        word = word_info['word']
        is_current = word_info.get('is_current', False)

        # Add space between words
        if i > 0:
            word = ' ' + word

        # Choose color based on whether word is currently being spoken
        color = config.get('highlight_color', (0, 255, 255)) if is_current else config['color']

        # Get text size
        (w, h), _ = cv2.getTextSize(word, font, font_scale, 2)

        # Render with outline
        render_text_with_outline(
            frame, word, (current_x, y), font, font_scale,
            color, config.get('outline_color'), config.get('outline_thickness', 2)
        )

        current_x += w


def render_caption_on_frame(frame, transcript_words, current_time, style_name="classic",
                            custom_color=None, custom_outline_color=None):
    """
    Main function to render captions on a video frame.

    Args:
        frame: OpenCV frame (numpy array)
        transcript_words: List of word dicts with 'word', 'start', 'end' keys
        current_time: Current timestamp in seconds
        style_name: Name of the caption style preset
        custom_color: Optional hex color for text (e.g., "#FFFFFF")
        custom_outline_color: Optional hex color for outline (e.g., "#000000")

    Returns:
        Frame with captions rendered
    """
    if not transcript_words:
        return frame

    # Get active words for this timestamp
    active_words = get_active_caption_text(transcript_words, current_time)

    if not active_words:
        return frame

    # Get style configuration
    config = get_style_config(style_name, custom_color, custom_outline_color)

    # Frame dimensions
    height, width = frame.shape[:2]

    # Font settings
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = config['font_scale'] * (width / 1080)  # Scale based on width

    # Build caption text
    caption_text = ' '.join(w['word'] for w in active_words)

    # Apply text transformations
    if config.get('uppercase'):
        caption_text = caption_text.upper()
    elif config.get('lowercase'):
        caption_text = caption_text.lower()

    # Calculate text size and position
    max_text_width = int(width * 0.9)  # 90% of frame width
    lines = wrap_text(caption_text, max_text_width, font, font_scale, 2)

    # Calculate total text block height
    line_heights = []
    for line in lines:
        (w, h), baseline = cv2.getTextSize(line, font, font_scale, 2)
        line_heights.append(h + baseline + 10)

    total_height = sum(line_heights)

    # Position at bottom of frame (10% from bottom)
    y_start = int(height * 0.85) - total_height

    # Render background box if style requires it
    if config.get('background'):
        bg_color = config['background']
        padding = 15

        # Calculate max line width
        max_line_width = 0
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, 2)
            max_line_width = max(max_line_width, w)

        # Draw semi-transparent background
        overlay = frame.copy()
        x1 = (width - max_line_width) // 2 - padding
        y1 = y_start - padding
        x2 = (width + max_line_width) // 2 + padding
        y2 = y_start + total_height + padding

        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color[:3], -1)
        alpha = bg_color[3] / 255.0 if len(bg_color) > 3 else 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Render each line
    current_y = y_start
    for line in lines:
        (w, h), baseline = cv2.getTextSize(line, font, font_scale, 2)
        x = (width - w) // 2  # Center horizontally
        current_y += h

        if style_name == "karaoke":
            # For karaoke, we need to render word by word
            # Filter active_words to only those in this line
            line_words = [w for w in active_words if w['word'] in line]
            render_karaoke_text(frame, line_words, (x, current_y), font, font_scale, config)
        elif style_name == "neon" and config.get('glow'):
            # Render glow effect
            for i in range(3, 0, -1):
                glow_color = tuple(min(255, c + 50) for c in config['color'])
                cv2.putText(frame, line, (x, current_y), font, font_scale,
                           glow_color, config['outline_thickness'] + i * 2, cv2.LINE_AA)
            render_text_with_outline(frame, line, (x, current_y), font, font_scale,
                                    config['color'], config.get('outline_color'),
                                    config.get('outline_thickness', 2))
        else:
            # Standard rendering with outline
            render_text_with_outline(frame, line, (x, current_y), font, font_scale,
                                    config['color'], config.get('outline_color'),
                                    config.get('outline_thickness', 2))

        current_y += baseline + 10

    return frame


def extract_words_from_transcript(transcript_result):
    """
    Extract flat list of words from transcript result.

    Args:
        transcript_result: Dict with 'segments' containing word-level data

    Returns:
        List of word dicts with 'word', 'start', 'end' keys
    """
    words = []

    if not transcript_result:
        return words

    segments = transcript_result.get('segments', [])

    for segment in segments:
        segment_words = segment.get('words', [])
        for word in segment_words:
            words.append({
                'word': word.get('word', ''),
                'start': word.get('start', 0),
                'end': word.get('end', 0)
            })

    return words


# List of available styles for CLI choices
AVAILABLE_STYLES = list(CAPTION_STYLES.keys())
