"""Deterministic rule-based + fuzzy-matching parser (works cleanly with zero API keys)."""

from __future__ import annotations

import re

from rapidfuzz import fuzz, process

from dfv.search.schema import PropertyFilters, check_query_safety

AREA_ALIASES: dict[str, str] = {
    "jvc": "Jumeirah Village Circle",
    "jlt": "Jumeirah Lake Towers",
    "jvt": "Jumeirah Village Triangle",
    "marina": "Dubai Marina",
    "dubai marina": "Dubai Marina",
    "downtown": "Downtown Dubai",
    "downtown dubai": "Downtown Dubai",
    "palm": "Palm Jumeirah",
    "palm jumeirah": "Palm Jumeirah",
    "business bay": "Business Bay",
    "creek harbour": "Dubai Creek Harbour",
    "dubai creek": "Dubai Creek Harbour",
    "dch": "Dubai Creek Harbour",
    "dubai hills": "Dubai Hills Estate",
    "dhe": "Dubai Hills Estate",
    "arabian ranches": "Arabian Ranches",
    "ranches": "Arabian Ranches",
    "al furjan": "Al Furjan",
    "furjan": "Al Furjan",
    "discovery gardens": "Discovery Gardens",
    "discovery": "Discovery Gardens",
    "dubai south": "Dubai South",
    "city walk": "City Walk",
    "beachfront": "Emaar Beachfront",
    "emaar beachfront": "Emaar Beachfront",
    "damac hills": "Damac Hills",
    "sports city": "Dubai Sports City",
    "silicon oasis": "Dubai Silicon Oasis",
    "dso": "Dubai Silicon Oasis",
    "international city": "International City",
    "dip": "Dubai Investment Park First",
    "motor city": "Motor City",
    "arjan": "Arjan",
    "majan": "Majan",
    "sobha hartland": "MBR City / Sobha Hartland",
    "mbr city": "MBR City / Sobha Hartland",
}


def _scale_price(num_str: str, unit: str | None) -> float:
    val = float(num_str.replace(",", ""))
    if unit:
        u = unit.lower().strip()
        if u in {"m", "million", "mn"}:
            return val * 1_000_000.0
        if u == "k":
            return val * 1_000.0
    if val < 100.0:
        # Shorthand e.g. "under 1.5" in million-AED context when unit omitted alongside "budget"/"price"
        return val * 1_000_000.0
    if val < 10_000.0:
        return val * 1_000.0
    return val


def _extract_areas(query_lower: str, known_areas: list[str] | None) -> list[str]:
    found: list[str] = []
    # 1. Explicit curated aliases first (longest match priority)
    for alias in sorted(AREA_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", query_lower):
            canonical = AREA_ALIASES[alias]
            if canonical not in found:
                # Prefer exact known area if known_areas subset supplied
                if not known_areas or canonical in known_areas:
                    found.append(canonical)

    if known_areas:
        # 2. Direct substring scan against known areas
        for area in known_areas:
            if area.lower() in query_lower and area not in found:
                found.append(area)

        # 3. Fuzzy n-gram scan only when neither alias nor exact substring hit
        if not found:
            words = re.findall(r"[a-z0-9]+", query_lower)
            stop_words = {
                "bed",
                "bedroom",
                "bedrooms",
                "under",
                "below",
                "over",
                "above",
                "near",
                "metro",
                "good",
                "high",
                "yield",
                "with",
                "flat",
                "unit",
                "villa",
                "townhouse",
                "apartment",
                "off",
                "offplan",
                "plan",
                "ready",
                "resale",
                "aed",
                "dhs",
                "price",
                "within",
                "from",
                "for",
                "in",
                "to",
                "of",
                "at",
                "and",
                "the",
                "km",
                "sqm",
            }
            content_words = [
                w for w in words if w not in stop_words and not w.isdigit() and len(w) >= 3
            ]
            ngrams: list[str] = []
            for n in (3, 2, 1):
                for i in range(0, max(0, len(content_words) - n + 1)):
                    cand = " ".join(content_words[i : i + n])
                    if len(cand) >= 4:
                        ngrams.append(cand)
            best_score = 0.0
            best_match: str | None = None
            alias_keys = [
                k for k in AREA_ALIASES if not known_areas or AREA_ALIASES[k] in known_areas
            ]
            for ng in ngrams:
                res_k = process.extractOne(ng, known_areas, scorer=fuzz.WRatio, score_cutoff=85)
                if res_k and float(res_k[1]) > best_score:
                    best_score = float(res_k[1])
                    best_match = str(res_k[0])
                res_a = process.extractOne(ng, alias_keys, scorer=fuzz.ratio, score_cutoff=84)
                if res_a and float(res_a[1]) > best_score:
                    best_score = float(res_a[1])
                    best_match = AREA_ALIASES[str(res_a[0])]
            if best_match:
                found.append(best_match)

    return found[:10]


def parse_query_rules(
    query: str,
    known_areas: list[str] | None = None,
) -> PropertyFilters:
    """Deterministically parse natural-language property queries into validated PropertyFilters."""
    cleaned = check_query_safety(query)
    q = cleaned.lower()

    # Bedrooms
    rooms: int | None = None
    if re.search(r"\bstudio\b", q):
        rooms = 0
    else:
        m_bed = re.search(r"\b(\d+)\s*(?:-|\s)?(?:bedrooms?|beds?|b/?r|bhk)\b", q)
        if m_bed:
            rooms = int(m_bed.group(1))

    # Property type
    property_type: str | None = None
    if re.search(r"\b(?:villas?|townhouses?)\b", q):
        property_type = "Villa"
    elif re.search(r"\b(?:apartments?|apts?|flats?|units?|penthouses?|studios?)\b", q):
        property_type = "Unit"

    # Off-plan vs ready
    off_plan: bool | None = None
    if re.search(r"\boff[-\s]?plan\b|under\s+construction", q):
        off_plan = True
    elif re.search(r"\b(?:ready|resale|completed)\b", q):
        off_plan = False

    # Metro proximity
    near_metro_m: float | None = None
    m_metro_dist = re.search(
        r"(?:within|under|<=|<|max)\s*(\d+(?:\.\d+)?)\s*(km|m)\s*(?:from|of|to)?\s*(?:the\s*)?metro",
        q,
    )
    if m_metro_dist:
        dist_val = float(m_metro_dist.group(1))
        near_metro_m = dist_val * 1000.0 if m_metro_dist.group(2) == "km" else dist_val
    elif re.search(
        r"\b(?:near|close\s+to|by|walking\s+distance\s+to)\s+(?:a\s+|the\s+)?metro\b", q
    ):
        near_metro_m = 800.0

    # Yield
    min_yield_pct: float | None = None
    m_yield = re.search(
        r"(?:>=?|above|over|min(?:imum)?|at\s+least|with)?\s*(\d+(?:\.\d+)?)\s*%\s*(?:gross\s*)?yield|"
        r"yield\s*(?:>=?|above|over|of|at\s+least)\s*(\d+(?:\.\d+)?)\s*%?",
        q,
    )
    if m_yield:
        val_s = m_yield.group(1) or m_yield.group(2)
        if val_s:
            min_yield_pct = float(val_s)
    elif re.search(r"\b(?:good|high|strong)\s+(?:rental\s+)?yield\b", q):
        min_yield_pct = 6.5

    # Price bounds (strip metro/yield phrases temporarily or require price keywords/units)
    max_price_aed: float | None = None
    min_price_aed: float | None = None
    m_max_p = re.search(
        r"(?:under|below|max(?:imum)?|less\s+than|up\s+to|budget\s*(?:of)?|<=|<)\s*(?:aed|dhs)?\s*(\d[\d,]*(?:\.\d+)?)\s*(m|million|mn|k)?\b(?!(?:\s*(?:m|km)\s*(?:from|of|to)?\s*metro|\s*%|\s*bed))",
        q,
    )
    if m_max_p:
        val_max = _scale_price(m_max_p.group(1), m_max_p.group(2))
        if val_max >= 50_000.0:
            max_price_aed = val_max

    m_min_p = re.search(
        r"(?:over|above|min(?:imum)?|more\s+than|from|at\s+least|>=|>)\s*(?:aed|dhs)?\s*(\d[\d,]*(?:\.\d+)?)\s*(m|million|mn|k)?\b(?!(?:\s*(?:m|km)\s*(?:from|of|to)?\s*metro|\s*%|\s*bed|\s*sqm))",
        q,
    )
    if m_min_p:
        val_min = _scale_price(m_min_p.group(1), m_min_p.group(2))
        if val_min >= 50_000.0:
            min_price_aed = val_min

    # Size bounds (sqm)
    min_area_sqm: float | None = None
    max_area_sqm: float | None = None
    m_min_a = re.search(r"(?:over|above|min(?:imum)?|>=?)\s*(\d+(?:\.\d+)?)\s*sqm\b", q)
    if m_min_a:
        min_area_sqm = float(m_min_a.group(1))
    m_max_a = re.search(r"(?:under|below|max(?:imum)?|<=?)\s*(\d+(?:\.\d+)?)\s*sqm\b", q)
    if m_max_a:
        max_area_sqm = float(m_max_a.group(1))

    areas = _extract_areas(q, known_areas)

    # Swap inverted min/max gracefully if ambiguous parsing triggered both
    if min_price_aed is not None and max_price_aed is not None and min_price_aed > max_price_aed:
        min_price_aed = None

    return PropertyFilters(
        rooms=rooms,
        property_type=property_type,  # type: ignore[arg-type]
        min_price_aed=min_price_aed,
        max_price_aed=max_price_aed,
        min_area_sqm=min_area_sqm,
        max_area_sqm=max_area_sqm,
        areas=areas,
        near_metro_m=near_metro_m,
        min_yield_pct=min_yield_pct,
        off_plan=off_plan,
        limit=25,
    )
