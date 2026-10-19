"""Scopus search definitions: one query per electrocatalytic reaction, plus API and retry settings.

Run ``python -m mofcat.ingestion.queries`` to print the final queries.
"""


SCOPUS_URL = "https://api.elsevier.com/content/search/scopus"
SERIAL_URL = "https://api.elsevier.com/content/serial/title"

PAGE_SIZE = 25
MAX_RESULTS = None
SLEEP_SECONDS = 2.0
FETCH_ALL = True
MAX_RETRIES = 6

MOF_KEYWORDS = (
    'MOF OR MOFs OR "metal organic framework" OR '
    '"metal organic frameworks" OR '
    '"metal-organic framework" OR '
    '"metal-organic frameworks" OR '
    '"π-d"'
)

YEAR_FILTER = "(PUBYEAR > 2015 AND PUBYEAR < 2026)"

# ------------------------------------------------------------
# Open access filter — pick one:
# ------------------------------------------------------------
OA_FILTER = "OPENACCESS(1)"                                    # any open access
# OA_FILTER = "OA(publisherfullgold OR publisherhybridgold)"   # Gold + Hybrid only
# OA_FILTER = "OA(repository)"                                 # Green only
# OA_FILTER = None                                             # no OA filtering


SCOPUS_QUERIES = {

    "NO3RR": {
        "application": "Ammonia synthesis",
        "query": (
            f'TITLE-ABS-KEY({MOF_KEYWORDS}) '
            'AND TITLE-ABS-KEY('
            '"nitrate reduction to ammonia" OR '
            '"Nitrate reduction" OR '
            '"Nitrate electroreduction" OR '
            '"*NO3RR" OR '
            '"nitrate electrocatalysis" OR '
            '"NO3*" OR '
            '"nitrate" OR '
            '"nitrates"'
            ') '
            'AND TITLE-ABS-KEY('
            '"ammonia electrosynthesis" OR '
            '"ammonia production" OR '
            '"ammonia synthesis" OR '
            '"ammonia"'
            ') '
            f'AND {YEAR_FILTER}'
        )
    },

    "N2RR": {
        "application": "Ammonia synthesis",
        "query": (
            f'TITLE-ABS-KEY({MOF_KEYWORDS}) '
            'AND TITLE-ABS-KEY('
            '"nitrogen reduction to ammonia" OR '
            '"Nitrogen reduction" OR '
            '"Nitrogen fixation" OR '
            '"Nitrogen electrocatalysis" OR '
            '"Nitrogen electroreduction"'
            ') '
            'AND TITLE-ABS-KEY('
            '"ammonia electrosynthesis" OR '
            '"ammonia production" OR '
            '"ammonia synthesis" OR '
            '"Ammonia"'
            ') '
            f'AND {YEAR_FILTER}'
        )
    },

    "ORR": {
        "application": "Metal-air batteries",
        "query": (
            f'TITLE-ABS-KEY({MOF_KEYWORDS}) '
            'AND TITLE('
            '"oxygen reduction reaction" OR '
            "ORR" ' OR '
            ' "oxygen electrocatalysis" OR '
            ' "oxygen electroreduction" '
            ') '
            f'AND {YEAR_FILTER}'
        )
    },

    "HER": {
        "application": "Hydrogen evolution",
        "query": (
            f'TITLE-ABS-KEY({MOF_KEYWORDS}) '
            'AND TITLE('
            '"hydrogen evolution reaction" OR '
            "HER"
            ') '
            f'AND {YEAR_FILTER}'
        )
    },

    "OER": {
        "application": "Metal-air batteries",
        "query": (
            f'TITLE-ABS-KEY({MOF_KEYWORDS}) '
            'AND TITLE('
            '"oxygen evolution reaction" OR '
            "OER" ' OR '
            ' "oxygen electrocatalysis" OR '
            ' "oxygen electroreduction" OR '
            ' "Oxygen Evolution" '
            ') '
            f'AND {YEAR_FILTER}'
        )
    }
}


# ------------------------------------------------------------
# Append the open access filter to every query
# ------------------------------------------------------------
if OA_FILTER:
    for _key, _cfg in SCOPUS_QUERIES.items():
        _cfg["query"] = f'({_cfg["query"]}) AND {OA_FILTER}'

# Canonical reaction order used by every later stage (also the sheet names in the Scopus workbook).
REACTIONS = ["HER", "OER", "ORR", "N2RR", "NO3RR"]


if __name__ == "__main__":
    # Quick check: print the final queries
    for name, cfg in SCOPUS_QUERIES.items():
        print(f"\n[{name}] {cfg['application']}\n{cfg['query']}")
