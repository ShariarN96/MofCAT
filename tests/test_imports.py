"""Smoke tests: every module imports and the shared paths/config are consistent. No network or API keys needed."""

import importlib

import pytest

MODULES = [
    "mofcat",
    "mofcat.paths",
    "mofcat.env",
    "mofcat.ingestion.queries",
    "mofcat.ingestion.fulltext",
    "mofcat.ingestion.wiley_tdm",
    "mofcat.ingestion.xml_to_markdown",
    "mofcat.ingestion.xml_to_text",
    "mofcat.extraction.vocabulary",
    "mofcat.extraction.reaction_schemas",
    "mofcat.extraction.data_checks",
    "mofcat.extraction.mof_verification",
    "mofcat.extraction.metrics_extract",
    "mofcat.extraction.linker_resolver",
    "mofcat.reporting.figures",
    "mofcat.reporting.science_figures",
    "mofcat.reporting.panel_figures",
    "mofcat.reporting.excel_export",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_every_reaction_has_a_query_and_schema():
    from mofcat.extraction.reaction_schemas import METRICS
    from mofcat.ingestion.queries import REACTIONS, SCOPUS_QUERIES

    assert set(REACTIONS) == set(SCOPUS_QUERIES) == set(METRICS)


def test_paths_live_under_project_root():
    from mofcat import paths

    for p in (paths.DATA_DIR, paths.OUTPUT_DIR, paths.SCOPUS_XLSX, paths.FULLTEXT_DIR):
        assert paths.PROJECT_ROOT in p.parents
    assert (paths.PROJECT_ROOT / "pyproject.toml").exists()
