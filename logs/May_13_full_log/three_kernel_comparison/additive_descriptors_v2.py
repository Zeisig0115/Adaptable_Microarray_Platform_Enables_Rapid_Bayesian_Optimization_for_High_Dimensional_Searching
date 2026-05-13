"""FAMT-targeted chemistry descriptors for the 22 additives.

This file is the DRAFT to be reviewed by the domain expert before being moved
into the FAMT kernel module. It is independent of `add_bo_mod.py`'s
ADDITIVE_METADATA (which is kept intact for the hier_family baseline).

Feature columns (13):

    Structural / class flags (7)
    ----------------------------
      is_peg               : 1 if the molecule is a polyethylene glycol
                             (linear PEG chain). PL127 (PEG-PPG-PEG block)
                             counts as 1 here.
      is_polymer           : 1 if MW indicates a polymer (>= ~1 kDa repeated
                             units). PEG400 still gets 1 to match the
                             existing convention.
      is_surfactant        : 1 if amphiphilic and forms micelles in water.
      is_polyol_sugar      : 1 if a small polyol or sugar (glycerol, sucrose).
      is_protein           : 1 if a protein.
      is_solvent           : 1 if an aprotic / miscible cosolvent.
      is_salt              : 1 if dissolves into metal cation + anion.

    Mechanism-specific flags for HRP / TMB / H2O2 (4)
    -------------------------------------------------
      is_redox_active      : 1 if the metal cation participates in
                             single-electron transfer with H2O2 (Fenton-type
                             chemistry). Fe2+ and Mn2+ score 1; Mg/Ca/Zn are 0.
      is_protein_stabilizer: 1 if a known osmolyte / crowder / shield that
                             extends HRP folding stability (BSA, glycerol,
                             sucrose, PEGs, PVA).
      is_chelator_strong   : 1 if forms strong complexes with the heme Fe and
                             thereby inhibits HRP at modest concentrations.
                             EDTA is the only 1 in this set; imidazole's
                             weak coordination is captured through charge
                             + low log_MW instead.
      is_kosmotrope_anion  : 1 if the salt's anion is kosmotropic on the
                             Hofmeister series (SO4^2- in this dataset).
                             Cl- is treated as borderline / 0.

    Continuous descriptors (2)
    --------------------------
      log_mw               : log10 of the active species' molecular weight
                             (in Daltons; polymer average MW for polymers).
      charge_at_pH7        : signed approximate charge of the dominant
                             species at pH 7. For salts, this is the metal
                             cation's charge (e.g., Mg2+ -> +2). For
                             polyanionic polymers (CMC, PAA) a saturated
                             value of -3 represents "strongly anionic".
                             For zwitterionic / partial-protonation species
                             (imidazole, pKa ~6.95) a partial value is used.

NOTE for the domain expert:
  - Entries flagged with `# ?` are best-guess and should be checked.
  - `is_protein_stabilizer` for surfactants is currently 0 (they can be
    weak stabilizers at low conc, but the dominant mechanism is different);
    flip this if you think otherwise.
  - `charge_at_pH7` for BSA is set to -3 as a coarse "slightly anionic at
    pH 7" proxy; BSA's pI is ~4.7 so it carries net negative charge above.
  - is_polymer == 1 for peg400 is kept to match existing convention; if
    you want oligomers separated, flip it.
"""

import math

# Keys, in the order they should be turned into columns of W.
FEATURE_KEYS = [
    # structural
    "is_peg",
    "is_polymer",
    "is_surfactant",
    "is_polyol_sugar",
    "is_protein",
    "is_solvent",
    "is_salt",
    # mechanism
    "is_redox_active",
    "is_protein_stabilizer",
    "is_chelator_strong",
    "is_kosmotrope_anion",
    # continuous
    "log_mw",
    "charge_at_pH7",
]

ADDITIVE_METADATA_V2: dict[str, dict[str, float]] = {
    "cmc": dict(
        is_peg=0, is_polymer=1, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(90000.0),
        charge_at_pH7=-3.0,  # ? anionic polysaccharide (carboxymethyl groups)
    ),
    "peg20k": dict(
        is_peg=1, is_polymer=1, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(20000.0),
        charge_at_pH7=0.0,
    ),
    "dmso": dict(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=1, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(78.13),
        charge_at_pH7=0.0,
    ),
    "pl127": dict(
        is_peg=1,             # PEG-PPG-PEG triblock => "has PEG character"
        is_polymer=1, is_surfactant=1, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,  # ? typical at low conc
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(12600.0),
        charge_at_pH7=0.0,
    ),
    "bsa": dict(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=1, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(66430.0),
        charge_at_pH7=-3.0,  # ? pI ~4.7, net negative at pH 7
    ),
    "pva": dict(
        is_peg=0, is_polymer=1, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(31000.0),
        charge_at_pH7=0.0,
    ),
    "tw80": dict(
        is_peg=0, is_polymer=0, is_surfactant=1, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(1310.0),
        charge_at_pH7=0.0,
    ),
    "glycerol": dict(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=1,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(92.09),
        charge_at_pH7=0.0,
    ),
    "tw20": dict(
        is_peg=0, is_polymer=0, is_surfactant=1, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(1228.0),
        charge_at_pH7=0.0,
    ),
    "imidazole": dict(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0,                 # weak coordinator only
        is_kosmotrope_anion=0,
        log_mw=math.log10(68.08),
        charge_at_pH7=0.5,  # pKa ~6.95 -> roughly half-protonated at pH 7
    ),
    "tx100": dict(
        is_peg=0,             # PEG chain in the headgroup but not classed as "PEG"
        is_polymer=0, is_surfactant=1, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(647.0),
        charge_at_pH7=0.0,
    ),
    "edta": dict(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=1,                 # the only strong chelator
        is_kosmotrope_anion=0,
        log_mw=math.log10(292.24),
        charge_at_pH7=-3.0,  # ? mostly EDTA^3- at pH 7 (pKa4 ~10.2)
    ),
    "mgso4": dict(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=1,
        is_redox_active=0,                    # Mg2+ not redox-active
        is_protein_stabilizer=0,
        is_chelator_strong=0,
        is_kosmotrope_anion=1,                # SO4^2- kosmotropic
        log_mw=math.log10(120.37),
        charge_at_pH7=2.0,                    # Mg2+
    ),
    "sucrose": dict(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=1,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(342.30),
        charge_at_pH7=0.0,
    ),
    "cacl2": dict(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=1,
        is_redox_active=0,                    # Ca2+ not redox-active
        is_protein_stabilizer=0,
        is_chelator_strong=0,
        is_kosmotrope_anion=0,                # Cl- chaotropic-ish
        log_mw=math.log10(110.98),
        charge_at_pH7=2.0,                    # Ca2+
    ),
    "znso4": dict(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=1,
        is_redox_active=0,                    # Zn2+ d10, not redox-active
        is_protein_stabilizer=0,
        is_chelator_strong=0,
        is_kosmotrope_anion=1,
        log_mw=math.log10(161.44),
        charge_at_pH7=2.0,                    # Zn2+
    ),
    "paa": dict(
        is_peg=0, is_polymer=1, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0,                 # weak metal binder, not strong
        is_kosmotrope_anion=0,
        log_mw=math.log10(100000.0),
        charge_at_pH7=-3.0,  # ? strongly anionic at pH 7 (pKa ~4.5)
    ),
    "mncl2": dict(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=1,
        is_redox_active=1,                    # Mn2+ <-> Mn3+ accessible
        is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(125.84),
        charge_at_pH7=2.0,                    # Mn2+
    ),
    "peg200k": dict(
        is_peg=1, is_polymer=1, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(200000.0),
        charge_at_pH7=0.0,
    ),
    "feso4": dict(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=1,
        is_redox_active=1,                    # Fe2+ Fenton chemistry with H2O2
        is_protein_stabilizer=0,
        is_chelator_strong=0,
        is_kosmotrope_anion=1,
        log_mw=math.log10(151.91),
        charge_at_pH7=2.0,                    # Fe2+
    ),
    "peg6k": dict(
        is_peg=1, is_polymer=1, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(6000.0),
        charge_at_pH7=0.0,
    ),
    "peg400": dict(
        is_peg=1, is_polymer=1,               # kept = 1 for compatibility
        is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(400.0),
        charge_at_pH7=0.0,
    ),
}


def build_descriptor_matrix(additives):
    """Return (W, FEATURE_KEYS) where W has shape (len(additives), len(FEATURE_KEYS))."""
    import numpy as np
    rows = []
    for name in additives:
        if name not in ADDITIVE_METADATA_V2:
            raise KeyError(f"Missing metadata for additive '{name}'.")
        rows.append([float(ADDITIVE_METADATA_V2[name][k]) for k in FEATURE_KEYS])
    return np.asarray(rows, dtype=float), list(FEATURE_KEYS)
