"""Canonical vocabularies for normalizing extracted MOF chemistry.

LLMs return chemically correct but textually inconsistent labels ("copper",
"Cu(II)", "Cu2+" all mean the same node metal). Everything is mapped onto a
controlled vocabulary here so that downstream counts and figures aggregate
correctly.
"""

from __future__ import annotations

import re

ELEMENT_NAMES: dict[str, str] = {
    "lithium": "Li", "sodium": "Na", "potassium": "K", "magnesium": "Mg",
    "calcium": "Ca", "strontium": "Sr", "barium": "Ba", "scandium": "Sc",
    "titanium": "Ti", "vanadium": "V", "chromium": "Cr", "manganese": "Mn",
    "iron": "Fe", "cobalt": "Co", "nickel": "Ni", "copper": "Cu",
    "zinc": "Zn", "gallium": "Ga", "germanium": "Ge", "yttrium": "Y",
    "zirconium": "Zr", "niobium": "Nb", "molybdenum": "Mo", "technetium": "Tc",
    "ruthenium": "Ru", "rhodium": "Rh", "palladium": "Pd", "silver": "Ag",
    "cadmium": "Cd", "indium": "In", "tin": "Sn", "antimony": "Sb",
    "lanthanum": "La", "cerium": "Ce", "praseodymium": "Pr", "neodymium": "Nd",
    "samarium": "Sm", "europium": "Eu", "gadolinium": "Gd", "terbium": "Tb",
    "dysprosium": "Dy", "holmium": "Ho", "erbium": "Er", "thulium": "Tm",
    "ytterbium": "Yb", "lutetium": "Lu", "hafnium": "Hf", "tantalum": "Ta",
    "tungsten": "W", "rhenium": "Re", "osmium": "Os", "iridium": "Ir",
    "platinum": "Pt", "gold": "Au", "mercury": "Hg", "lead": "Pb",
    "bismuth": "Bi", "thorium": "Th", "uranium": "U", "aluminum": "Al",
    "aluminium": "Al", "silicon": "Si", "beryllium": "Be",
}

VALID_SYMBOLS: set[str] = set(ELEMENT_NAMES.values())

# Linker aliases -> canonical family label. Keys are matched case-insensitively
# after light punctuation normalization.
LINKER_ALIASES: dict[str, str] = {
    # Benzene-1,4-dicarboxylate family
    "bdc": "BDC (terephthalate)",
    "h2bdc": "BDC (terephthalate)",
    "1,4-bdc": "BDC (terephthalate)",
    "tpa": "BDC (terephthalate)",
    "terephthalate": "BDC (terephthalate)",
    "terephthalic acid": "BDC (terephthalate)",
    "1,4-benzenedicarboxylate": "BDC (terephthalate)",
    "1,4-benzenedicarboxylic acid": "BDC (terephthalate)",
    "benzene-1,4-dicarboxylate": "BDC (terephthalate)",
    "nh2-bdc": "NH2-BDC (aminoterephthalate)",
    "2-aminoterephthalate": "NH2-BDC (aminoterephthalate)",
    "amino-bdc": "NH2-BDC (aminoterephthalate)",
    # Benzene-1,3,5-tricarboxylate family
    "btc": "BTC (trimesate)",
    "h3btc": "BTC (trimesate)",
    "1,3,5-btc": "BTC (trimesate)",
    "trimesate": "BTC (trimesate)",
    "trimesic acid": "BTC (trimesate)",
    "benzenetricarboxylate": "BTC (trimesate)",
    "1,3,5-benzenetricarboxylate": "BTC (trimesate)",
    "benzene-1,3,5-tricarboxylate": "BTC (trimesate)",
    # Imidazolates
    "2-methylimidazole": "Imidazolate (2-MIM)",
    "2-methylimidazolate": "Imidazolate (2-MIM)",
    "2-mim": "Imidazolate (2-MIM)",
    "hmim": "Imidazolate (2-MIM)",
    "mim": "Imidazolate (2-MIM)",
    "imidazolate": "Imidazolate",
    "imidazole": "Imidazolate",
    "benzimidazolate": "Benzimidazolate",
    "benzimidazole": "Benzimidazolate",
    # Extended dicarboxylates
    "bpdc": "BPDC (biphenyldicarboxylate)",
    "biphenyldicarboxylate": "BPDC (biphenyldicarboxylate)",
    "4,4'-biphenyldicarboxylate": "BPDC (biphenyldicarboxylate)",
    "ndc": "NDC (naphthalenedicarboxylate)",
    "naphthalenedicarboxylate": "NDC (naphthalenedicarboxylate)",
    "2,6-ndc": "NDC (naphthalenedicarboxylate)",
    # Tri/tetratopic carboxylates
    "btb": "BTB (benzenetribenzoate)",
    "benzenetribenzoate": "BTB (benzenetribenzoate)",
    "bptc": "BPTC (biphenyltetracarboxylate)",
    "btec": "BTEC (pyromellitate)",
    "pyromellitate": "BTEC (pyromellitate)",
    "tcpp": "TCPP (porphyrin)",
    "tcpp porphyrin": "TCPP (porphyrin)",
    "tetrakis(4-carboxyphenyl)porphyrin": "TCPP (porphyrin)",
    "porphyrin": "Porphyrin",
    "phthalocyanine": "Phthalocyanine",
    "tbapy": "TBAPy (pyrene)",
    "pyrene": "TBAPy (pyrene)",
    # N-donor linkers
    "bipyridine": "Bipyridine (bpy)",
    "bpy": "Bipyridine (bpy)",
    "4,4'-bipyridine": "Bipyridine (bpy)",
    "bpydc": "BPyDC (bipyridinedicarboxylate)",
    "triazolate": "Azolate (triazolate)",
    "tetrazolate": "Azolate (tetrazolate)",
    "pyrazolate": "Azolate (pyrazolate)",
    "bdp": "Azolate (pyrazolate)",
    "adenine": "Adenine",
    "melamine": "Melamine",
    # Conductive / catecholate-type
    "hhtp": "HHTP (triphenylene)",
    "hexahydroxytriphenylene": "HHTP (triphenylene)",
    "2,3,6,7,10,11-hexahydroxytriphenylene": "HHTP (triphenylene)",
    "hitp": "HITP (hexaiminotriphenylene)",
    "hexaiminotriphenylene": "HITP (hexaiminotriphenylene)",
    "bht": "BHT (benzenehexathiol)",
    "benzenehexathiol": "BHT (benzenehexathiol)",
    "thq": "THQ (tetrahydroxyquinone)",
    "catecholate": "Catecholate",
    "dobdc": "DOBDC (dihydroxyterephthalate)",
    "dhbdc": "DOBDC (dihydroxyterephthalate)",
    "2,5-dihydroxyterephthalate": "DOBDC (dihydroxyterephthalate)",
    "dhta": "DOBDC (dihydroxyterephthalate)",
    # Aliphatic / small linkers
    "fumarate": "Fumarate",
    "fumaric acid": "Fumarate",
    "formate": "Formate",
    "acetate": "Acetate",
    "oxalate": "Oxalate",
    "succinate": "Succinate",
    "malonate": "Malonate",
    "citrate": "Citrate",
    # Other
    "phosphonate": "Phosphonate",
    "sulfonate": "Sulfonate",
    "salen": "Salen",
    "salophen": "Salen",
    "tpt": "TPT (triazine)",
    "triazine": "TPT (triazine)",
    "nicotinate": "Nicotinate",
    "isonicotinate": "Isonicotinate",
    "trimellitate": "Trimellitate",
    "ta": "Tannic acid",
    "tannic acid": "Tannic acid",
}

# Known MOF names -> (canonical family, typical node metal, typical linker).
# Used to backfill chemistry when an abstract names the MOF but not its parts.
MOF_FAMILY_DEFAULTS: dict[str, tuple[str, str | None, str | None]] = {
    "zif-8": ("ZIF", "Zn", "Imidazolate (2-MIM)"),
    "zif-67": ("ZIF", "Co", "Imidazolate (2-MIM)"),
    "zif-9": ("ZIF", "Co", "Benzimidazolate"),
    "zif-12": ("ZIF", "Co", "Benzimidazolate"),
    "zif-7": ("ZIF", "Zn", "Benzimidazolate"),
    "uio-66": ("UiO", "Zr", "BDC (terephthalate)"),
    "uio-67": ("UiO", "Zr", "BPDC (biphenyldicarboxylate)"),
    "uio-68": ("UiO", "Zr", "BPDC (biphenyldicarboxylate)"),
    "mil-53": ("MIL", "Al", "BDC (terephthalate)"),
    "mil-88": ("MIL", "Fe", "BDC (terephthalate)"),
    "mil-100": ("MIL", "Fe", "BTC (trimesate)"),
    "mil-101": ("MIL", "Cr", "BDC (terephthalate)"),
    "mil-125": ("MIL", "Ti", "BDC (terephthalate)"),
    "mil-127": ("MIL", "Fe", "BPTC (biphenyltetracarboxylate)"),
    "hkust-1": ("HKUST", "Cu", "BTC (trimesate)"),
    "mof-5": ("MOF-5/IRMOF", "Zn", "BDC (terephthalate)"),
    "irmof-1": ("MOF-5/IRMOF", "Zn", "BDC (terephthalate)"),
    "mof-74": ("MOF-74/CPO-27", None, "DOBDC (dihydroxyterephthalate)"),
    "cpo-27": ("MOF-74/CPO-27", None, "DOBDC (dihydroxyterephthalate)"),
    "mof-808": ("MOF-808", "Zr", "BTC (trimesate)"),
    "nu-1000": ("NU", "Zr", "TBAPy (pyrene)"),
    "pcn-222": ("PCN", "Zr", "TCPP (porphyrin)"),
    "pcn-224": ("PCN", "Zr", "TCPP (porphyrin)"),
    "mof-525": ("MOF-525/545", "Zr", "TCPP (porphyrin)"),
    "mof-545": ("MOF-525/545", "Zr", "TCPP (porphyrin)"),
    "cu3(hhtp)2": ("Conductive 2D MOF", "Cu", "HHTP (triphenylene)"),
    "ni3(hitp)2": ("Conductive 2D MOF", "Ni", "HITP (hexaiminotriphenylene)"),
    "prussian blue": ("Prussian blue analogue", "Fe", "Cyanide"),
    "pba": ("Prussian blue analogue", None, "Cyanide"),
}

MATERIAL_TYPES = {"pristine_mof", "mof_derived", "mof_composite", "not_mof", "unknown"}

REACTIONS = ["HER", "OER", "ORR", "N2RR", "NO3RR"]

REACTION_ALIASES: dict[str, str] = {
    "her": "HER", "hydrogen evolution": "HER", "hydrogen evolution reaction": "HER",
    "oer": "OER", "oxygen evolution": "OER", "oxygen evolution reaction": "OER",
    "orr": "ORR", "oxygen reduction": "ORR", "oxygen reduction reaction": "ORR",
    "n2rr": "N2RR", "nrr": "N2RR", "nitrogen reduction": "N2RR",
    "nitrogen reduction reaction": "N2RR", "n2 reduction": "N2RR",
    "no3rr": "NO3RR", "nitrate reduction": "NO3RR", "no3- reduction": "NO3RR",
    "nitrate reduction reaction": "NO3RR", "narr": "NO3RR",
}

_OXIDATION_STATE = re.compile(r"\s*\((?:0|i{1,3}v?|vi{0,3}|[0-9])\)\s*$", re.I)
_CHARGE_SUFFIX = re.compile(r"\s*\d*[+-]$")


def normalize_metal(raw: str) -> str | None:
    """Map a metal mention to its element symbol, or None if unrecognized."""
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None

    text = _OXIDATION_STATE.sub("", text)
    text = _CHARGE_SUFFIX.sub("", text).strip()
    text = re.sub(r"[-\s]*(?:based|containing|doped|site|centers?|centres?|node|ion)s?$", "", text, flags=re.I)
    text = text.strip(" -–—.,;:")

    if not text:
        return None
    if text in VALID_SYMBOLS:
        return text
    if text.capitalize() in VALID_SYMBOLS:
        return text.capitalize()

    lowered = text.lower()
    if lowered in ELEMENT_NAMES:
        return ELEMENT_NAMES[lowered]

    # Handle "Ni-Fe" or "CoFe" bimetallics by taking the leading element only;
    # callers receive each metal separately so ordering is preserved upstream.
    parts = re.split(r"[-/,\s]+", lowered)
    for part in parts:
        if part in ELEMENT_NAMES:
            return ELEMENT_NAMES[part]
        if part.capitalize() in VALID_SYMBOLS:
            return part.capitalize()
    return None


def normalize_linker(raw: str) -> str | None:
    """Map a linker mention onto the controlled linker vocabulary."""
    if not raw or not isinstance(raw, str):
        return None
    text = re.sub(r"\s+", " ", raw.strip().lower())
    text = text.strip(" -–—.,;:")
    if not text or text in {"none", "n/a", "na", "unknown", "not specified", "not reported"}:
        return None

    # Progressively simpler forms of the same mention. Fuller forms are tried
    # first so "terephthalic acid" resolves before it decays to "terephthalic".
    candidates = [text]
    deprotonated = re.sub(r"^h\d+(?=[a-z])", "", text)  # H2BDC -> BDC
    if deprotonated != text:
        candidates.append(deprotonated)
    for candidate in list(candidates):
        stripped = re.sub(r"\s*(?:linker|ligand|acid|anion)s?$", "", candidate).strip()
        if stripped and stripped != candidate:
            candidates.append(stripped)

    for candidate in candidates:
        if candidate in LINKER_ALIASES:
            return LINKER_ALIASES[candidate]

    # Longest-alias substring match keeps "2-methylimidazolate" from matching
    # the shorter "imidazolate" alias.
    for alias in sorted(LINKER_ALIASES, key=len, reverse=True):
        if len(alias) >= 4 and any(alias in candidate for candidate in candidates):
            return LINKER_ALIASES[alias]

    return raw.strip()[:60]


def normalize_mof_name(raw: str) -> tuple[str | None, str | None, str | None]:
    """Return (family, default_metal, default_linker) for a named MOF.

    An explicit metal in the name ("MIL-101(Fe)", "MOF-74-Ni") overrides the
    family's default node metal, since these isoreticular series are routinely
    reported with substituted metals.
    """
    if not raw or not isinstance(raw, str):
        return None, None, None
    text = re.sub(r"\s+", "", raw.strip().lower()).replace("–", "-").replace("_", "-")

    explicit_metal: str | None = None
    parenthetical = re.search(r"\(([a-z]{1,12})\)", text)
    if parenthetical:
        explicit_metal = normalize_metal(parenthetical.group(1))
    if not explicit_metal:
        trailing = re.search(r"-([a-z]{1,12})$", text)
        if trailing:
            explicit_metal = normalize_metal(trailing.group(1))

    def result(entry: tuple[str, str | None, str | None]) -> tuple[str | None, str | None, str | None]:
        family, metal, linker = entry
        return family, explicit_metal or metal, linker

    if text in MOF_FAMILY_DEFAULTS:
        return result(MOF_FAMILY_DEFAULTS[text])

    # Match a numbered family member such as "zif-67(co)" or "mil-101(fe)".
    match = re.match(r"^([a-z]+(?:-\d+)?)", text)
    if match and match.group(1) in MOF_FAMILY_DEFAULTS:
        return result(MOF_FAMILY_DEFAULTS[match.group(1)])

    prefix = re.match(r"^(zif|uio|mil|pcn|nu|mof|hkust|nott|dut|cau|iso)", text)
    if prefix:
        label = prefix.group(1).upper()
        family = {"UIO": "UiO", "MOF": "MOF-other"}.get(label, label)
        return family, explicit_metal, None
    return None, explicit_metal, None


def normalize_reaction(raw: str) -> str | None:
    """Map a reaction mention to one of the five target reaction labels."""
    if not raw or not isinstance(raw, str):
        return None
    text = re.sub(r"\s+", " ", raw.strip().lower()).strip(" -–—.,;:")
    if not text:
        return None
    if text.upper() in REACTIONS:
        return text.upper()
    if text in REACTION_ALIASES:
        return REACTION_ALIASES[text]
    for alias, canonical in sorted(REACTION_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if alias in text:
            return canonical
    return None


def normalize_material_type(raw: str) -> str:
    """Coerce a material-type label into the controlled set."""
    if not raw or not isinstance(raw, str):
        return "unknown"
    text = re.sub(r"[\s-]+", "_", raw.strip().lower())
    if text in MATERIAL_TYPES:
        return text
    if "derive" in text or "pyroly" in text or "carboniz" in text:
        return "mof_derived"
    if "composite" in text or "hybrid" in text:
        return "mof_composite"
    if "pristine" in text or "intact" in text or text == "mof":
        return "pristine_mof"
    if "not" in text and "mof" in text:
        return "not_mof"
    return "unknown"
