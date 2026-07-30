import re

def clean_stylometric_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Strip Markdown Formatting & Layout Bullets
    text = re.sub(r'\*\*|__', '', text)  # Removes bold markers
    text = re.sub(r'#+\s+', '', text)    # Removes header hashes
    
    # NEW: Strips leading list dashes or bullet points at the start of lines
    text = re.sub(r'(?m)^\s*[-\*\+]\s+', '', text) 
    
    # Safely remove standalone asterisks or underscores without breaking hyphens
    text = re.sub(r'[*_]', '', text)     
    
    # 2. Strip Volatile Marketing Buzzwords safely
    buzzwords = [
        "groundbreaking", "empowering", "revolutionize", "testament",
        "delve", "foster", "tapestry", "subsequent", "pivotal", "seamlessly"
    ]
    for word in buzzwords:
        text = re.sub(r'\b' + word + r'\b', ' ', text, flags=re.IGNORECASE)
        
    # 3. Clean up spacing clusters
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text