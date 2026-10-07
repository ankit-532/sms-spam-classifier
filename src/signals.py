# src/signals.py
# Structural Signal Detectors - EXACT same as Colab

import re

def detect_premium_number(text):
    """Detect UK premium-rate numbers: 09xxxxxxxxx or 087xxxxxxx"""
    return bool(re.search(r'\b09\d{9}\b', text)) or bool(re.search(r'\b087\d{7,8}\b', text))

def detect_shortcode(text):
    """Detect standalone 4-6 digit shortcodes"""
    return bool(re.search(r'\b\d{4,6}\b', text))

def detect_stop_language(text):
    """Detect STOP/unsubscribe language"""
    pattern = r'(txt|text|send|reply)\s*stop|stop\s*2\s*stop|stop\?'
    return bool(re.search(pattern, text, re.IGNORECASE))

def get_structural_signals(text):
    """Return dict of all structural signals detected in the message"""
    return {
        "premium_number": detect_premium_number(text),
        "shortcode": detect_shortcode(text),
        "stop_language": detect_stop_language(text)
    }