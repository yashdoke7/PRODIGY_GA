import re

def clean_text(text):
    return re.sub(r'[^A-Za-z0-9\s\.\,\!\?]', '', text).lower().strip()

