"""Per-reaction extraction schemas (HER, OER, ORR, N2RR, NO3RR).

Each paper yields a list of records; one record = one catalyst under one set of
measurement conditions. Metrics are declared as (name, type, description) so the
Pydantic models and the prompt are generated from a single source of truth.
Edit METRICS to add or drop a metric; bump PROMPT_VERSION to invalidate the cache.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, create_model

PROMPT_VERSION = "elsevier-v3"

# type tags: float | str | bool | list
FieldSpec = tuple[str, str, str]

COMMON_FIELDS: list[FieldSpec] = [
    ("catalyst_name", "str", "Catalyst name exactly as written (e.g. 'Ni-BDC/NF', 'CoFe-MOF-P')."),
    ("role", "str", "main = the paper's featured/best catalyst; variant = other catalysts, ratios or "
                    "controls made by the authors; benchmark = commercial or literature reference "
                    "(Pt/C, RuO2, IrO2, ...)."),
    ("material_class", "str", "pristine_mof = a MOF used as synthesized, including MOF nanosheets/arrays grown "
                              "on a substrate | mof_derived = made BY pyrolysis, calcination, phosphidation, "
                              "sulfidation etc. of a MOF precursor | mof_composite = MOF combined with another "
                              "component (nanoparticles, graphene, ...) | non_mof | unclear"),
    ("mof_or_precursor_name", "str", "Named MOF the catalyst is, or was derived from (e.g. 'ZIF-67', 'Ni-BDC', "
                                     "'MIL-101(Fe)'); null if none is named."),
    ("node_metals", "list", "Framework/precursor metals as element symbols (all metals for multimetallic materials)."),
    ("organic_linkers", "list", "Organic linkers/ligands NAMED IN THE TEXT, as written (e.g. '2-methylimidazole', "
                                "'terephthalic acid (H2BDC)', 'trimesic acid'). Leave empty if the text does not name "
                                "the linker; do not infer it from the MOF name."),
    ("morphology", "str", "Morphology as described by the authors (nanosheet, nanorod array, hollow polyhedron, "
                          "flower-like, nanoparticle, thin film, ...)."),
    ("crystallinity", "str", "crystalline | semi-crystalline | amorphous | not stated"),
    ("bet_surface_area_m2_g", "float", "BET specific surface area, m2 g-1 (number only)."),
    ("synthesis_method", "str", "Synthesis route in a few words (solvothermal, hydrothermal, room-temperature "
                               "stirring, electrodeposition, pyrolysis at 800 C, phosphidation, ...)."),
    ("electrolyte", "str", "Electrolyte for this measurement, as written (e.g. '1.0 M KOH')."),
    ("medium", "str", "acidic | neutral | alkaline | other"),
    ("reference_electrode", "str", "Reference scale of the potentials in this record (RHE, SHE, Ag/AgCl, ...)."),
    ("substrate_or_electrode", "str", "Support or working electrode (nickel foam, GCE, carbon cloth, ...)."),
    ("cell_type", "str", "H-cell, flow cell, MEA, RDE/RRDE, coin cell, ..."),
    ("stability_duration_h", "float", "Longest stability test duration in hours."),
    ("stability_cycles", "float", "Number of cycles in a cycling stability test."),
    ("stability_retention_pct", "float", "Activity or efficiency retained after the stability test, percent."),
    ("stability_condition", "str", "Current density or potential held during the stability test, with unit "
                                  "(e.g. '10 mA cm-2', '-0.3 V vs RHE')."),
    ("stability_note", "str", "Stability result in words, e.g. 'potential increased by 12 mV after 100 h'."),
]

METRICS: dict[str, list[FieldSpec]] = {
    "HER": [
        ("overpotential_10_ma_cm2_mv", "float", "Overpotential (mV) at 10 mA cm-2."),
        ("overpotential_100_ma_cm2_mv", "float", "Overpotential (mV) at 100 mA cm-2."),
        ("tafel_slope_mv_dec", "float", "Tafel slope, mV dec-1."),
        ("exchange_current_density", "float", "Exchange current density j0, number only."),
        ("exchange_current_density_unit", "str", "Unit of j0 exactly as reported."),
        ("tof_value", "float", "Turnover frequency, number only."),
        ("tof_unit", "str", "TOF unit as reported (s-1, ...)."),
        ("tof_at_overpotential_mv", "float", "Overpotential at which TOF was reported."),
        ("mass_activity_value", "float", "Mass activity, number only."),
        ("mass_activity_unit", "str", "Mass activity unit as reported."),
        ("mass_activity_at_overpotential_mv", "float", "Overpotential at which mass activity was reported."),
        ("cdl_mf_cm2", "float", "Double-layer capacitance Cdl, mF cm-2."),
        ("ecsa_value", "float", "Electrochemically active surface area, number only."),
        ("ecsa_unit", "str", "ECSA unit as reported."),
        ("rct_ohm", "float", "Charge-transfer resistance Rct from EIS, ohm."),
        ("overall_water_splitting_voltage_v", "float", "Cell voltage (V) at 10 mA cm-2 for overall water splitting."),
    ],
    "OER": [
        ("overpotential_10_ma_cm2_mv", "float", "Overpotential (mV) at 10 mA cm-2."),
        ("overpotential_100_ma_cm2_mv", "float", "Overpotential (mV) at 100 mA cm-2."),
        ("tafel_slope_mv_dec", "float", "Tafel slope, mV dec-1."),
        ("tof_value", "float", "Turnover frequency, number only."),
        ("tof_unit", "str", "TOF unit as reported."),
        ("tof_at_overpotential_mv", "float", "Overpotential at which TOF was reported."),
        ("mass_activity_value", "float", "Mass activity, number only."),
        ("mass_activity_unit", "str", "Mass activity unit as reported."),
        ("mass_activity_at_overpotential_mv", "float", "Overpotential at which mass activity was reported."),
        ("cdl_mf_cm2", "float", "Double-layer capacitance Cdl, mF cm-2."),
        ("ecsa_value", "float", "Electrochemically active surface area, number only."),
        ("ecsa_unit", "str", "ECSA unit as reported."),
        ("rct_ohm", "float", "Charge-transfer resistance Rct from EIS, ohm."),
        ("faradaic_efficiency_o2_pct", "float", "Faradaic efficiency toward O2, percent."),
        ("overall_water_splitting_voltage_v", "float", "Cell voltage (V) at 10 mA cm-2 for overall water splitting."),
        ("surface_reconstruction_observed", "bool", "True if the paper reports in-situ reconstruction of the MOF "
                                                    "into (oxy)hydroxide or another active phase."),
    ],
    "ORR": [
        ("onset_potential_v", "float", "Onset potential (V) on the scale given in reference_electrode."),
        ("half_wave_potential_v", "float", "Half-wave potential E1/2 (V) on the scale given in reference_electrode."),
        ("limiting_current_density_ma_cm2", "float", "Diffusion-limited current density, mA cm-2 (absolute value)."),
        ("electron_transfer_number", "float", "Electron transfer number n from K-L or RRDE."),
        ("h2o2_yield_pct", "float", "Peroxide (H2O2 / HO2-) yield, percent."),
        ("kinetic_current_density_ma_cm2", "float", "Kinetic current density jk, mA cm-2."),
        ("kinetic_current_density_at_potential_v", "float", "Potential at which jk was reported."),
        ("tafel_slope_mv_dec", "float", "Tafel slope, mV dec-1."),
        ("mass_activity_value", "float", "Mass activity, number only."),
        ("mass_activity_unit", "str", "Mass activity unit as reported."),
        ("mass_activity_at_potential_v", "float", "Potential at which mass activity was reported."),
        ("methanol_tolerance", "str", "Methanol crossover tolerance result in words or percent retained."),
        ("zn_air_peak_power_density_mw_cm2", "float", "Zn-air battery peak power density, mW cm-2."),
        ("zn_air_specific_capacity_mah_g", "float", "Zn-air specific capacity, mAh g-1."),
        ("zn_air_cycle_life_h", "float", "Zn-air battery cycling life, hours."),
    ],
    "N2RR": [
        ("potential_v", "float", "Potential at which this record's NH3 numbers were measured."),
        ("nh3_yield_rate", "float", "NH3 production rate, number only."),
        ("nh3_yield_rate_unit", "str", "Unit exactly as reported (ug h-1 mg-1cat, umol h-1 cm-2, ...)."),
        ("nh3_faradaic_efficiency_pct", "float", "Faradaic efficiency toward NH3, percent."),
        ("total_current_density_ma_cm2", "float", "Total current density, mA cm-2."),
        ("nh3_partial_current_density_ma_cm2", "float", "NH3 partial current density, mA cm-2."),
        ("n2h4_detected", "bool", "True if hydrazine was tested for and detected."),
        ("n2h4_faradaic_efficiency_pct", "float", "Faradaic efficiency toward N2H4, percent."),
        ("isotope_15n2_labeling", "bool", "True if a 15N2 isotope labeling experiment is reported."),
        ("blank_controls_reported", "bool", "True if Ar / open-circuit / no-catalyst controls are reported."),
        ("nh3_quantification_method", "str", "indophenol blue, Nessler, NMR, ion chromatography, ..."),
    ],
    "NO3RR": [
        ("potential_v", "float", "Potential at which this record's NH3 numbers were measured."),
        ("nh3_yield_rate", "float", "NH3 production rate, number only."),
        ("nh3_yield_rate_unit", "str", "Unit exactly as reported (mg h-1 cm-2, umol h-1 mg-1cat, ...)."),
        ("nh3_faradaic_efficiency_pct", "float", "Faradaic efficiency toward NH3, percent."),
        ("nh3_selectivity_pct", "float", "NH3 selectivity, percent."),
        ("nitrate_conversion_pct", "float", "Nitrate conversion, percent."),
        ("initial_nitrate_concentration", "str", "Initial nitrate concentration with unit, as reported."),
        ("total_current_density_ma_cm2", "float", "Total current density, mA cm-2."),
        ("nh3_partial_current_density_ma_cm2", "float", "NH3 partial current density, mA cm-2."),
        ("nitrite_selectivity_or_fe_pct", "float", "NO2- selectivity or Faradaic efficiency, percent."),
        ("n2_selectivity_or_fe_pct", "float", "N2 selectivity or Faradaic efficiency, percent."),
        ("isotope_15n_labeling", "bool", "True if a 15NO3- isotope labeling experiment is reported."),
        ("nh3_quantification_method", "str", "indophenol blue, Nessler, NMR, ion chromatography, ..."),
        ("zn_nitrate_battery_peak_power_density_mw_cm2", "float", "Zn-nitrate battery peak power density, mW cm-2."),
    ],
}

TAIL_FIELDS: list[FieldSpec] = [
    ("source_location", "str", "Where the numbers came from: text | table | figure_caption | abstract."),
    ("evidence_quote", "str", "Verbatim snippet (<=200 chars) supporting the main number(s) of this record."),
]

REACTION_GUIDANCE: dict[str, str] = {
    "HER": "Hydrogen evolution reaction. Report overpotential at the stated current density; do not "
           "convert between current densities.",
    "OER": "Oxygen evolution reaction. Overpotential is relative to 1.23 V vs RHE; if the paper gives "
           "potentials instead, report only what is stated, do not compute it.",
    "ORR": "Oxygen reduction reaction. Potentials are on the scale in reference_electrode; do not "
           "convert between scales unless the paper itself states the converted value.",
    "N2RR": "Electrochemical nitrogen reduction to ammonia. Report the best yield/FE and, as separate "
            "records, other potentials if a table lists them. Capture isotope-labeling and control experiments.",
    "NO3RR": "Electrochemical nitrate reduction to ammonia. Report NH3 yield/FE/selectivity with the "
             "potential and nitrate concentration they were measured at.",
}

_PY_TYPES = {"float": float | None, "str": str | None, "bool": bool | None}


def _field_definitions(specs: list[FieldSpec]) -> dict[str, Any]:
    defs: dict[str, Any] = {}
    for name, kind, desc in specs:
        if kind == "list":
            defs[name] = (list[str], Field(default_factory=list, description=desc))
        else:
            defs[name] = (_PY_TYPES[kind], Field(None, description=desc))
    return defs


def record_specs(reaction: str) -> list[FieldSpec]:
    return COMMON_FIELDS + METRICS[reaction] + TAIL_FIELDS


def metric_names(reaction: str) -> list[str]:
    return [n for n, _, _ in METRICS[reaction]]


def build_paper_model(reaction: str):
    """Return the Pydantic model the LLM fills for one paper of this reaction."""
    record_model = create_model(f"{reaction}Record", **_field_definitions(record_specs(reaction)))
    return create_model(
        f"{reaction}Paper",
        is_experimental_study=(bool, Field(
            description="True only if the paper reports NEW experimental electrochemical data "
                        "(False for reviews, perspectives, purely computational papers).")),
        reaction_measured=(bool, Field(
            description=f"True if the paper measures {reaction} performance for a catalyst.")),
        is_mof_related=(bool, Field(
            description="True if the catalyst is a MOF, MOF composite or MOF-derived material.")),
        mof_names=(list[str], Field(default_factory=list, description="MOF names used in the paper.")),
        records=(list[record_model], Field(default_factory=list)),
        confidence=(float, Field(ge=0.0, le=1.0, description="Overall extraction confidence 0-1.")),
        notes=(str | None, Field(None, description="Anything ambiguous or unusual.")),
    )


def describe_fields(reaction: str) -> str:
    return "\n".join(f"- {n}: {d}" for n, _, d in record_specs(reaction))


SYSTEM_TEMPLATE = """You are an expert electrocatalysis information extractor building a structured \
database of MOF and MOF-derived catalysts from full-text journal articles.

Reaction: {reaction}. {guidance}

You receive the full article text: abstract, body, tables (rows separated by ' | ') and figure \
captions. References have been removed. Superscripts appear as ^x and subscripts as _x.

Output one record per catalyst per measurement condition set. Include the main catalyst, the \
catalyst variants and controls the authors prepared, and benchmark materials (Pt/C, RuO2, IrO2, \
...) when they have reported numbers. Set 'role' accordingly.

Rules:
- Extract ONLY values explicitly stated in the text, tables or captions. Never estimate from \
figures, never compute, never convert units or potential scales. Use null when a value is not stated.
- Copy numbers as reported; put units in the *_unit fields exactly as written.
- Prefer table values; they are usually the cleanest. Literature-comparison tables list catalysts \
from OTHER papers (rows with citation numbers, 'Ref.', or a 'This work' row among them): extract only \
the paper's own rows ('This work') and skip the rest.
- If several potentials/conditions are tabulated for the same catalyst, create one record each, \
but do not exceed 12 records per paper; keep the main catalyst's best-performing conditions first.
- A paper with no new experimental {reaction} data (review, theory only) gets \
is_experimental_study=false or reaction_measured=false and an empty records list.
- Stability fields describe one catalyst's long-term test. Fill them on that catalyst's records \
only from its own stability statement; do not reuse numbers from unrelated measurements.
- Keep 'catalyst_name' consistent for the same material across records.
- Material descriptors (organic_linkers, morphology, crystallinity, BET area, synthesis_method, \
mof_or_precursor_name) are properties of the catalyst, not of the measurement: repeat them on every \
record of that catalyst, taking them from the synthesis and characterization sections.
- Linkers: copy the names the authors use. If the paper never names the linker, return an empty list.
- evidence_quote must be verbatim from the text.

Record fields:
{fields}"""

HUMAN_TEMPLATE = """TITLE: {title}
DOI: {doi}
YEAR: {year}

FULL TEXT:
{text}

Return the structured extraction."""


def system_prompt(reaction: str) -> str:
    return SYSTEM_TEMPLATE.format(
        reaction=reaction,
        guidance=REACTION_GUIDANCE[reaction],
        fields=describe_fields(reaction),
    )
