"""
preprocess.py - Text normalization utilities for business entity resolution.
"""

import re
import pandas as pd

# Abbreviation expansion maps
ABBR_NAME = {
    "corp": "corporation",
    "pvt": "private",
    "ltd": "limited",
    "llc": "limited liability company",
    "inc": "incorporated",
    "&": "and",
    "co": "company",
    "intl": "international",
    "mfg": "manufacturing",
    "mgmt": "management",
    "grp": "group",
    "assoc": "association",
    "svc": "services",
    "svcs": "services",
    "dept": "department",
    "natl": "national",
    "fedl": "federal",
    "govt": "government",
    "univ": "university",
    "hosp": "hospital",
    "med": "medical",
    "tech": "technology",
    "sys": "systems",
    "dev": "development",
    "mkt": "marketing",
    "fin": "financial",
    "ins": "insurance",
    "bldg": "building",
    "ctr": "center",
    "cnt": "county",
}

ABBR_ADDR = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "apt": "apartment",
    "no": "number",
    "dist": "district",
    "opp": "opposite",
    "nr": "near",
    "hwy": "highway",
    "ln": "lane",
    "dr": "drive",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "expy": "expressway",
    "fwy": "freeway",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "nw": "northwest",
    "ne": "northeast",
    "sw": "southwest",
    "se": "southeast",
}


def normalize(text, abbr_map: dict) -> str:
    """
    Normalize a text string:
    - Convert to lowercase
    - Strip leading/trailing whitespace
    - Remove all non-word, non-space characters
    - Expand abbreviations using abbr_map
    - Collapse multiple spaces to single space

    Returns empty string for NaN / None / non-string inputs.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""

    text = text.lower().strip()
    # Remove all characters that are not word chars (\w = [a-z0-9_]) or spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Tokenize
    tokens = text.split()
    # Expand abbreviations
    expanded = []
    for tok in tokens:
        expanded.append(abbr_map.get(tok, tok))
    # Rejoin and collapse spaces
    return " ".join(expanded).strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clean_name and clean_address columns to a copy of df.
    """
    df = df.copy()
    df["clean_name"] = df["business_name"].apply(lambda x: normalize(x, ABBR_NAME))
    df["clean_address"] = df["business_address"].apply(
        lambda x: normalize(x, ABBR_ADDR)
    )
    return df
