import os
import pandas as pd
import re

# paths
input_file = "data/nba_playoff_shots_2015_2024.csv"
output_dir = "cleaned_data"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "nba_playoff_2015_2024_structured.csv")

# data loading
df = pd.read_csv(input_file)

# fixing encoding issues
def fix_encoding(text):
    try:
        if isinstance(text, str) and 'Ã' in text:
            return text.encode('latin1').decode('utf-8')
        return text
    except:
        return text

df['desc'] = df['desc'].apply(fix_encoding)

# only keeping actual shot attempts
def is_shot_event(desc):
    if not isinstance(desc, str):
        return False
    phrases = ['makes', 'misses']
    return any(p in desc for p in phrases)

df = df[df['desc'].apply(is_shot_event)].reset_index(drop=True)

def extract_shooter(desc):
    m = re.match(r'^([A-Z]\. [A-Za-zÀ-ÿ\'\-]+)', desc)
    return m.group(1) if m else None

def extract_result(desc):
    if "makes" in desc:
        return "make"
    elif "misses" in desc:
        return "miss"
    return None

def extract_shot_type(desc):
    desc_lower = desc.lower()
    for t in ["free throw", "tip-in", "layup", "dunk", "hook", "floater", "step back", "fadeaway", "jump"]:
        if t in desc_lower:
            return t
    if "3-pt" in desc_lower and "jump" in desc_lower:
        return "3-pt jump"
    return "other"

def extract_distance(desc):
    m = re.search(r'from (\d+) ft', desc)
    return int(m.group(1)) if m else None

def extract_assist(desc):
    # Handles both with/without space after "assist by"
    m = re.search(r'assist by\s*([A-Z]\. [A-Za-zÀ-ÿ\'\-]+)', desc)
    return m.group(1) if m else "Unassisted"

def extract_block(desc):
    # Handles both with/without space after "block by"
    m = re.search(r'block by\s*([A-Z]\. [A-Za-zÀ-ÿ\'\-]+)', desc)
    return m.group(1) if m else "Not Blocked"

df['shooter'] = df['desc'].apply(extract_shooter)
df['shot_result'] = df['desc'].apply(extract_result)
df['shot_type'] = df['desc'].apply(extract_shot_type)
df['shot_distance'] = df['desc'].apply(extract_distance)
df['assisted_by'] = df['desc'].apply(extract_assist)
df['blocked_by'] = df['desc'].apply(extract_block)

final_cols = [
    'period', 'time_left', 'team', 'score', 'source_url',
    'shooter', 'shot_result', 'shot_type', 'shot_distance', 'assisted_by', 'blocked_by', 'desc'
]
df = df[final_cols]

df.to_csv(output_file, index=False)
print(df.head(10))
print(f"Cleaned and structured file saved as: {output_file}")

print(df[['desc', 'assisted_by', 'blocked_by']].head(20))
