"""
DooDoo Bio Weather System — Genomic & Climate Pattern Analysis
==============================================================
Same aoi_collapse() that trades markets, now reading biological
and climate data as "weather systems."

25 TE families provide the vocabulary. The collapse provides the perception.
DooDoo tells you what she sees.

Usage:
    python doodoo_bio_weather.py                  # Demo with climate data
    python doodoo_bio_weather.py --climate        # Analyze Climate-REF data
    python doodoo_bio_weather.py --synthetic      # Generate synthetic bio weather
    python doodoo_bio_weather.py --csv FILE.csv   # Analyze any CSV data
"""

import sys
import os
import numpy as np
from datetime import datetime
from aoi_collapse import aoi_collapse

# ============================================================
# TE CATALOG — 25 transposable element families as weather patterns
# Each TE maps to a data signature (chaos range, intent, control profile)
# ============================================================

TE_CATALOG = {
    # --- DNA Transposons (cut-and-paste / rolling-circle) ---
    "HELITRON": {
        "mechanism": "rolling-circle",
        "weather": "spiral vortex",
        "signature": {"chaos": (3.0, 7.0), "intent": "directional", "control": "rotational"},
        "desc": "Rolling-circle replication capturing neighboring signals — data spiraling inward",
        "bio": "Gene-capturing DNA transposon using helicase-driven strand displacement",
    },
    "CACTA": {
        "mechanism": "cut-and-paste",
        "weather": "cold front with debris",
        "signature": {"chaos": (4.0, 8.0), "intent": "strong", "control": "bimodal"},
        "desc": "Large transposon capturing host genes then going silent — spike then calm",
        "bio": "En/Spm superfamily with epigenetic silencing and gene capture",
    },
    "MUTATOR": {
        "mechanism": "cut-and-paste",
        "weather": "mutagenic squall",
        "signature": {"chaos": (5.0, 9.0), "intent": "diffuse", "control": "scattered"},
        "desc": "High-chaos gene capture with multi-directional insertion — scattered storms",
        "bio": "Mu/MULE transposons with Pack-MULE gene capture and host silencing",
    },
    "CRYPTON": {
        "mechanism": "recombinase",
        "weather": "phase transition",
        "signature": {"chaos": (2.0, 5.0), "intent": "precise", "control": "sharp"},
        "desc": "Tyrosine recombinase making clean excision/integration — regime change",
        "bio": "YR-mediated transposon using site-specific recombination, not cut-and-paste",
    },
    "POLINTON": {
        "mechanism": "self-synthesizing",
        "weather": "self-organizing system",
        "signature": {"chaos": (3.0, 6.0), "intent": "strong", "control": "structured"},
        "desc": "Self-replicating element with own DNA polymerase — autonomous structure",
        "bio": "Giant transposon encoding its own replication machinery (DJR capsid + pPolB)",
    },
    "TRANSIB": {
        "mechanism": "cut-and-paste",
        "weather": "ancient foundation",
        "signature": {"chaos": (0.5, 3.0), "intent": "strong", "control": "stable"},
        "desc": "Domesticated transposase ancestor of immune system — deep stable pattern",
        "bio": "DDE transposase ancestor of RAG1, domesticated into V(D)J recombination",
    },

    # --- LTR Retrotransposons ---
    "TY3_GYPSY": {
        "mechanism": "LTR-retrotransposon",
        "weather": "echoing thunderstorm",
        "signature": {"chaos": (4.0, 8.0), "intent": "moderate", "control": "oscillating"},
        "desc": "Copy-paste amplification with periodic bursts — reverberating signal",
        "bio": "Chromodomain-containing LTR retrotransposon targeting heterochromatin",
    },
    "BEL_PAO": {
        "mechanism": "LTR-retrotransposon",
        "weather": "cross-ocean storm",
        "signature": {"chaos": (5.0, 9.0), "intent": "moderate", "control": "long-range"},
        "desc": "Horizontal gene transfer across kingdoms — cross-domain chaos event",
        "bio": "LTR retrotransposon with evidence of horizontal transfer across phyla",
    },
    "DIRS1": {
        "mechanism": "LTR-retrotransposon",
        "weather": "circular storm system",
        "signature": {"chaos": (3.0, 7.0), "intent": "moderate", "control": "circular"},
        "desc": "Tyrosine recombinase-based retrotransposon forming circular intermediates",
        "bio": "Uses YR instead of integrase, inverted LTRs, circular DNA intermediate",
    },
    "ERV_REACTIVATION": {
        "mechanism": "endogenous-retrovirus",
        "weather": "dormant volcano awakening",
        "signature": {"chaos": (6.0, 10.0), "intent": "explosive", "control": "sudden"},
        "desc": "Silenced viral element reactivating — sudden eruption from calm baseline",
        "bio": "Epigenetically silenced ERV reactivated by demethylation or stress",
    },
    "LTR_PALINDROME": {
        "mechanism": "LTR-retrotransposon",
        "weather": "mirrored pressure system",
        "signature": {"chaos": (2.0, 5.0), "intent": "symmetric", "control": "reflected"},
        "desc": "Palindromic LTR structures creating symmetric solo LTR remnants",
        "bio": "LTR recombination creating solo LTRs with regulatory potential",
    },
    "MAVERICK": {
        "mechanism": "self-synthesizing",
        "weather": "rogue wave",
        "signature": {"chaos": (5.0, 9.0), "intent": "independent", "control": "unpredictable"},
        "desc": "Giant self-replicating elements with viral capsid genes — autonomous rogue",
        "bio": "Polinton-like element encoding DJR capsid protein, may form virions",
    },

    # --- Non-LTR Retrotransposons (LINEs/SINEs) ---
    "LINE": {
        "mechanism": "non-LTR-retrotransposon",
        "weather": "persistent drizzle",
        "signature": {"chaos": (3.0, 6.0), "intent": "weak", "control": "diffuse"},
        "desc": "Constant low-level copy-paste activity — background noise that accumulates",
        "bio": "L1 autonomous non-LTR retrotransposon with ORF1p/ORF2p machinery",
    },
    "L1_SOMATIC": {
        "mechanism": "somatic-retrotransposition",
        "weather": "localized flash flood",
        "signature": {"chaos": (6.0, 10.0), "intent": "localized", "control": "intense"},
        "desc": "Somatic LINE-1 insertions in specific tissues — concentrated local chaos",
        "bio": "L1 retrotransposition in somatic cells, especially tumors and neurons",
    },
    "L1_NEURONAL": {
        "mechanism": "neuronal-retrotransposition",
        "weather": "neural lightning storm",
        "signature": {"chaos": (5.0, 8.0), "intent": "focused", "control": "branching"},
        "desc": "L1 jumping in developing neurons creating mosaic diversity — branching paths",
        "bio": "L1 retrotransposition during neurogenesis creating somatic mosaicism",
    },
    "SINE": {
        "mechanism": "non-autonomous-retrotransposon",
        "weather": "parasitic fog",
        "signature": {"chaos": (1.0, 4.0), "intent": "weak", "control": "clinging"},
        "desc": "Non-autonomous elements hijacking LINE machinery — low-energy parasitism",
        "bio": "Short elements (Alu, B1, B2) using L1 ORF2p for retrotransposition",
    },
    "ALU_EXPANSION": {
        "mechanism": "SINE-expansion",
        "weather": "spreading haze",
        "signature": {"chaos": (2.0, 5.0), "intent": "expansive", "control": "diffuse"},
        "desc": "Alu elements spreading through primate genomes — gradual pervasive change",
        "bio": "Most abundant TE in humans, >1M copies, regulatory exaptation",
    },
    "SVA_REGULATORY": {
        "mechanism": "composite-retrotransposon",
        "weather": "regulatory pressure shift",
        "signature": {"chaos": (3.0, 6.0), "intent": "regulatory", "control": "modulating"},
        "desc": "Composite element rewiring gene regulation — pressure system reorganization",
        "bio": "SINE-VNTR-Alu composite creating new regulatory elements near genes",
    },
    "CR1_JOCKEY": {
        "mechanism": "non-LTR-retrotransposon",
        "weather": "tropical disturbance",
        "signature": {"chaos": (3.0, 7.0), "intent": "moderate", "control": "drifting"},
        "desc": "Ancient non-LTR lineage with long evolutionary persistence — drifting system",
        "bio": "CR1/Jockey clade retrotransposons found across vertebrates and insects",
    },
    "RTE": {
        "mechanism": "non-LTR-retrotransposon",
        "weather": "jet stream transfer",
        "signature": {"chaos": (5.0, 9.0), "intent": "directional", "control": "fast"},
        "desc": "Horizontal transfer between distant species — jet stream moving signals",
        "bio": "BovB/RTE elements with extensive HGT between vertebrates and invertebrates",
    },
    "INDO": {
        "mechanism": "non-LTR-retrotransposon",
        "weather": "targeted disruption",
        "signature": {"chaos": (4.0, 7.0), "intent": "targeted", "control": "precise"},
        "desc": "Site-specific insertion disrupting host genes — precision strike",
        "bio": "Ingi/Dong elements with site-specific insertion disrupting VSG genes",
    },
    "I_R_ELEMENT": {
        "mechanism": "non-LTR-retrotransposon",
        "weather": "oscillating equilibrium",
        "signature": {"chaos": (2.0, 5.0), "intent": "balanced", "control": "oscillating"},
        "desc": "Copy number equilibrium with site-specific insertion — balanced system",
        "bio": "I-element retrotransposons with rDNA site specificity and copy number control",
    },
    "PENELOPE": {
        "mechanism": "PLE-retrotransposon",
        "weather": "deep ocean current",
        "signature": {"chaos": (4.0, 8.0), "intent": "deep", "control": "slow"},
        "desc": "Ancient element with GIY-YIG endonuclease and HGT — deep persistent flow",
        "bio": "Penelope-like elements encoding RT + GIY-YIG, with widespread HGT",
    },
    "VIPER_NGARO": {
        "mechanism": "YR-retrotransposon",
        "weather": "chimeric supercell",
        "signature": {"chaos": (5.0, 9.0), "intent": "complex", "control": "multi-layered"},
        "desc": "YR retrotransposon with domain shuffling — complex multi-layered storm",
        "bio": "Tyrosine recombinase retrotransposons with diverse domain architectures",
    },

    # --- Domesticated / Functional ---
    "RAG_LIKE": {
        "mechanism": "domesticated-transposase",
        "weather": "clear skies (engineered)",
        "signature": {"chaos": (0.0, 2.0), "intent": "precise", "control": "structured"},
        "desc": "Transposase domesticated into immune system — purposeful, stable, clear",
        "bio": "RAG1/2 domesticated from Transib for V(D)J recombination in immunity",
    },
    "HERV_SYNAPSE": {
        "mechanism": "domesticated-retrovirus",
        "weather": "neural aurora",
        "signature": {"chaos": (2.0, 6.0), "intent": "connective", "control": "networked"},
        "desc": "Retroviral Arc protein enabling synaptic communication — aurora display",
        "bio": "Arc/dArc capsid from Ty3/gypsy Gag enabling intercellular RNA transfer at synapses",
    },
}


def match_te_pattern(chaos: float, intent_mag: float, control_vec: np.ndarray) -> list:
    """
    Match collapse output signature to TE weather patterns.
    Returns top 3 matching TE families sorted by fit quality.
    """
    control_norm = float(np.linalg.norm(control_vec))
    matches = []

    for te_name, te in TE_CATALOG.items():
        sig = te["signature"]
        chaos_lo, chaos_hi = sig["chaos"]

        # Chaos range fit (0-1, 1=perfect)
        if chaos_lo <= chaos <= chaos_hi:
            chaos_fit = 1.0 - abs(chaos - (chaos_lo + chaos_hi) / 2) / ((chaos_hi - chaos_lo) / 2)
        elif chaos < chaos_lo:
            chaos_fit = max(0, 1.0 - (chaos_lo - chaos) / 3.0)
        else:
            chaos_fit = max(0, 1.0 - (chaos - chaos_hi) / 3.0)

        # Intent profile fit
        intent_label = sig["intent"]
        if intent_label in ("strong", "explosive", "directional", "focused", "targeted"):
            intent_fit = min(1.0, intent_mag / 0.5)
        elif intent_label in ("weak", "diffuse", "clinging"):
            intent_fit = max(0, 1.0 - intent_mag / 0.5)
        elif intent_label in ("moderate", "balanced", "regulatory"):
            intent_fit = 1.0 - abs(intent_mag - 0.35) / 0.35
        elif intent_label in ("precise", "symmetric"):
            intent_fit = min(1.0, intent_mag / 0.3) if intent_mag < 0.5 else 0.8
        elif intent_label in ("expansive", "connective", "independent", "localized", "complex", "deep"):
            intent_fit = 0.5  # neutral
        else:
            intent_fit = 0.5

        intent_fit = max(0, intent_fit)

        # Control profile fit
        ctrl_label = sig["control"]
        ctrl_ratio = abs(control_vec[0]) / (control_norm + 1e-8) if len(control_vec) >= 1 else 0
        ctrl_spread = np.std(control_vec) if len(control_vec) >= 2 else 0

        if ctrl_label in ("stable", "structured", "sharp"):
            ctrl_fit = min(1.0, ctrl_ratio * 1.5)
        elif ctrl_label in ("scattered", "diffuse"):
            ctrl_fit = min(1.0, ctrl_spread / 2.0)
        elif ctrl_label in ("rotational", "circular", "oscillating"):
            ctrl_fit = 1.0 - ctrl_ratio  # low dominance = rotational
        elif ctrl_label in ("fast", "sudden", "intense"):
            ctrl_fit = min(1.0, control_norm / 3.0)
        else:
            ctrl_fit = 0.5

        ctrl_fit = max(0, ctrl_fit)

        # Weighted score
        score = chaos_fit * 0.5 + intent_fit * 0.3 + ctrl_fit * 0.2
        matches.append((te_name, score, te))

    matches.sort(key=lambda x: -x[1])
    return matches[:3]


def encode_climate_data(variables: dict, window: int = 24) -> np.ndarray:
    """
    Encode climate variables into a 24D state vector.

    variables: dict of {name: np.ndarray} where arrays are time series
    window: rolling window for feature extraction

    Returns 24D state vector suitable for aoi_collapse().
    """
    state = np.zeros(24)
    var_names = sorted(variables.keys())
    n_vars = len(var_names)
    if n_vars == 0:
        return state

    slots_per_var = max(1, 24 // n_vars)
    idx = 0

    for vi, vname in enumerate(var_names):
        series = np.asarray(variables[vname], dtype=np.float64).ravel()
        if len(series) == 0:
            continue

        # Normalize to z-scores
        mu, sigma = series.mean(), series.std()
        if sigma > 1e-12:
            z = (series - mu) / sigma
        else:
            z = series - mu

        base = vi * slots_per_var

        # Feature 1: Current value (z-score of last point)
        if base < 24:
            state[base] = z[-1]

        # Feature 2: Trend (slope of last window)
        if base + 1 < 24 and len(z) >= 3:
            w = min(window, len(z))
            x = np.arange(w)
            slope = np.polyfit(x, z[-w:], 1)[0]
            state[base + 1] = slope * 10  # scale up

        # Feature 3: Volatility (std of last window)
        if base + 2 < 24 and len(z) >= 3:
            w = min(window, len(z))
            state[base + 2] = np.std(z[-w:])

        # Feature 4: Acceleration (second derivative)
        if base + 3 < 24 and len(z) >= 5:
            w = min(window, len(z))
            x = np.arange(w)
            coeffs = np.polyfit(x, z[-w:], 2)
            state[base + 3] = coeffs[0] * 50  # curvature

        # Feature 5: Anomaly score (how far from normal)
        if base + 4 < 24:
            state[base + 4] = abs(z[-1])

        # Feature 6: Cross-correlation with next variable
        if base + 5 < 24 and n_vars >= 2:
            next_var = var_names[(vi + 1) % n_vars]
            other = np.asarray(variables[next_var], dtype=np.float64).ravel()
            if len(other) >= 3 and len(z) >= 3:
                minlen = min(len(z), len(other))
                corr = np.corrcoef(z[-minlen:], other[-minlen:])[0, 1]
                state[base + 5] = corr if np.isfinite(corr) else 0

    # Scale to norm range that produces meaningful chaos variation
    # Market data at norm ~3 gives chaos 0-10 naturally
    # Bio/climate data needs slightly higher norm to excite the octonion algebra
    norm = np.linalg.norm(state)
    if norm > 0.01:
        # More feature variance = slightly higher excitation
        # Keep in 3-6 range to avoid pegging at max chaos
        target_norm = 3.0 + min(np.std(state) * 2.0, 3.0)
        state = state / norm * target_norm

    return state


def encode_generic_csv(filepath: str) -> np.ndarray:
    """Encode a CSV file into 24D state vector. First column = index, rest = variables."""
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1, filling_values=0)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    variables = {}
    for col in range(1, min(data.shape[1], 7)):  # up to 6 variables
        variables[f"var_{col}"] = data[:, col]
    return encode_climate_data(variables)


def generate_weather_report(state: np.ndarray, label: str = "observation") -> dict:
    """
    Run collapse on encoded state and generate a full weather report.
    Returns dict with collapse results, TE matches, and narrative.
    """
    result = aoi_collapse(state)
    chaos = result['normalized_chaos']
    intent = result['intent_magnitude']
    control = result['control_vec']
    prompt = result['text_prompt_base']

    # Match TE patterns
    matches = match_te_pattern(chaos, intent, control)

    # Build narrative
    primary = matches[0]
    te_name, score, te_info = primary

    # Overall condition
    if chaos < 2:
        condition = "STABLE"
        emoji_word = "clear"
    elif chaos < 5:
        condition = "DEVELOPING"
        emoji_word = "shifting"
    elif chaos < 8:
        condition = "ACTIVE"
        emoji_word = "turbulent"
    else:
        condition = "SEVERE"
        emoji_word = "extreme"

    narrative = (
        f"[{condition}] {label}\n"
        f"  Chaos: {chaos:.1f}/10 | Intent: {intent:.3f} | Control norm: {np.linalg.norm(control):.2f}\n"
        f"  Primary pattern: {te_name} — \"{te_info['weather']}\"\n"
        f"    {te_info['desc']}\n"
        f"  Bio analog: {te_info['bio']}\n"
    )

    if len(matches) > 1:
        secondary = matches[1]
        narrative += f"  Secondary: {secondary[0]} — \"{secondary[2]['weather']}\" (fit={secondary[1]:.2f})\n"

    if len(matches) > 2:
        tertiary = matches[2]
        narrative += f"  Tertiary:  {tertiary[0]} — \"{tertiary[2]['weather']}\" (fit={tertiary[1]:.2f})\n"

    return {
        "collapse": result,
        "te_matches": matches,
        "condition": condition,
        "narrative": narrative,
    }


def load_climate_ref_data(zip_path: str) -> dict:
    """
    Load climate data from the Climate-REF Zenodo archive.
    Returns dict of {filename: {variable_name: array}}.
    """
    import zipfile
    import io

    datasets = {}

    outer = zipfile.ZipFile(zip_path)
    inner_name = [n for n in outer.namelist() if n.endswith('.zip')][0]
    inner_data = outer.read(inner_name)
    inner = zipfile.ZipFile(io.BytesIO(inner_data))

    nc_files = [n for n in inner.namelist() if n.endswith('.nc')]

    for ncf in nc_files:
        try:
            import netCDF4
            import tempfile

            data = inner.read(ncf)
            tmp = os.path.join(tempfile.gettempdir(), 'doodoo_climate_tmp.nc')
            with open(tmp, 'wb') as f:
                f.write(data)

            nc = netCDF4.Dataset(tmp)
            variables = {}
            for vname, var in nc.variables.items():
                arr = var[:]
                if hasattr(arr, 'filled'):
                    arr = arr.filled(np.nan)
                if arr.ndim >= 1 and arr.size > 1:
                    variables[vname] = arr.ravel()
            nc.close()
            os.remove(tmp)

            if variables:
                short_name = ncf.split('/')[-1]
                datasets[short_name] = variables

        except Exception:
            continue

    return datasets


def generate_synthetic_bio_weather(n_timepoints: int = 200) -> dict:
    """
    Generate synthetic multi-variable data mimicking biological/climate signals.
    Includes regime changes, coupling, and transposon-like burst events.
    """
    t = np.linspace(0, 20 * np.pi, n_timepoints)
    rng = np.random.RandomState(42)

    # Base signals with different frequencies (like different climate variables)
    temp = np.sin(t * 0.5) * 2 + rng.randn(n_timepoints) * 0.3
    pressure = np.cos(t * 0.3) * 1.5 + rng.randn(n_timepoints) * 0.2
    humidity = np.sin(t * 0.7 + 1.0) * 1.0 + rng.randn(n_timepoints) * 0.4
    wind = np.abs(np.sin(t * 0.2)) * 3.0 + rng.randn(n_timepoints) * 0.5

    # Add "transposon burst" events (sudden regime changes)
    for burst_center in [50, 120, 170]:
        burst_width = 10
        burst = np.exp(-0.5 * ((np.arange(n_timepoints) - burst_center) / burst_width) ** 2)
        temp += burst * rng.randn() * 5  # ERV reactivation-like
        humidity += burst * 3  # Helitron capture-like

    # Add "horizontal transfer" (cross-variable coupling shift)
    coupling_start = 90
    for i in range(coupling_start, n_timepoints):
        wind[i] += 0.3 * temp[i - 1]  # RTE-like transfer

    return {
        "temperature": temp,
        "pressure": pressure,
        "humidity": humidity,
        "wind_speed": wind,
    }


def run_rolling_analysis(variables: dict, window: int = 24, step: int = 5) -> list:
    """
    Run rolling window analysis over time series data.
    Returns list of weather reports at each step.
    """
    # Find shortest series
    min_len = min(len(v) for v in variables.values())
    reports = []

    for start in range(0, max(1, min_len - window), step):
        end = start + window
        windowed = {k: v[start:end] for k, v in variables.items() if len(v) >= end}
        if not windowed:
            break

        state = encode_climate_data(windowed, window=window)
        report = generate_weather_report(state, label=f"t={start}-{end}")
        report["time_start"] = start
        report["time_end"] = end
        reports.append(report)

    return reports


def print_weather_bulletin(reports: list, title: str = "DooDoo Bio Weather Bulletin"):
    """Print a formatted weather bulletin from rolling analysis."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Powered by aoi_collapse() — same brain that trades live markets")
    print(f"{'=' * 70}")

    # Summary stats
    chaos_vals = [r["collapse"]["normalized_chaos"] for r in reports]
    conditions = [r["condition"] for r in reports]

    print(f"\n  Overview: {len(reports)} observations")
    print(f"  Chaos range: {min(chaos_vals):.1f} - {max(chaos_vals):.1f}")
    print(f"  Conditions: {', '.join(f'{c}={conditions.count(c)}' for c in sorted(set(conditions)))}")

    # Find notable events
    print(f"\n  --- NOTABLE EVENTS ---")
    for idx, r in enumerate(reports):
        chaos = r["collapse"]["normalized_chaos"]
        if chaos > 5.0 or r["condition"] in ("ACTIVE", "SEVERE"):
            te = r["te_matches"][0]
            if "time_start" in r:
                loc = f"[{r['time_start']:3d}-{r['time_end']:3d}]"
            else:
                loc = f"[obs {idx:3d}]"
            print(f"  {loc} "
                  f"chaos={chaos:.1f} {r['condition']:8s} "
                  f"=> {te[0]} \"{te[2]['weather']}\"")

    # Dominant patterns
    print(f"\n  --- DOMINANT TE PATTERNS ---")
    te_counts = {}
    for r in reports:
        name = r["te_matches"][0][0]
        te_counts[name] = te_counts.get(name, 0) + 1
    for name, count in sorted(te_counts.items(), key=lambda x: -x[1])[:5]:
        pct = count / len(reports) * 100
        te = TE_CATALOG[name]
        print(f"  {name:20s} {count:3d} ({pct:4.1f}%)  \"{te['weather']}\"")

    # Regime transitions
    print(f"\n  --- REGIME TRANSITIONS ---")
    for i in range(1, len(reports)):
        prev_te = reports[i-1]["te_matches"][0][0]
        curr_te = reports[i]["te_matches"][0][0]
        if prev_te != curr_te:
            prev_chaos = reports[i-1]["collapse"]["normalized_chaos"]
            curr_chaos = reports[i]["collapse"]["normalized_chaos"]
            delta = curr_chaos - prev_chaos
            arrow = ">>>" if abs(delta) > 2 else "=>"
            t_label = reports[i].get('time_start', i)
            print(f"  [{t_label:3d}]: "
                  f"{prev_te} {arrow} {curr_te} "
                  f"(chaos {prev_chaos:.1f} -> {curr_chaos:.1f})")

    print(f"\n{'=' * 70}")


def run_climate_ref_analysis(zip_path: str):
    """Analyze Climate-REF data from Zenodo archive."""
    print("Loading Climate-REF data from Zenodo archive...")
    datasets = load_climate_ref_data(zip_path)
    print(f"Loaded {len(datasets)} datasets")

    if not datasets:
        print("No datasets loaded. Need netCDF4 library.")
        return

    # Analyze each dataset
    all_reports = []
    for name, variables in sorted(datasets.items())[:10]:  # top 10
        # Filter to numeric arrays with >1 element
        usable = {k: v for k, v in variables.items()
                  if v.ndim == 1 and v.size > 3 and not k.endswith('_bnds')}
        if not usable:
            continue

        state = encode_climate_data(usable)
        report = generate_weather_report(state, label=name)
        all_reports.append(report)
        print(report["narrative"])

    if all_reports:
        print_weather_bulletin(all_reports, "Climate-REF Analysis")


def run_demo():
    """Run demo with synthetic bio weather data."""
    print("Generating synthetic bio weather data...")
    variables = generate_synthetic_bio_weather(200)

    print(f"Variables: {list(variables.keys())}")
    for name, vals in variables.items():
        print(f"  {name}: {len(vals)} points, range [{vals.min():.2f}, {vals.max():.2f}]")

    # Rolling analysis
    reports = run_rolling_analysis(variables, window=24, step=5)
    print_weather_bulletin(reports, "Synthetic Bio Weather — DooDoo's Domain Expansion")

    # Single-shot analysis of full dataset
    print("\n\n--- FULL DATASET SNAPSHOT ---")
    state = encode_climate_data(variables)
    report = generate_weather_report(state, label="Full synthetic dataset")
    print(report["narrative"])


if __name__ == '__main__':
    args = sys.argv[1:]

    if '--climate' in args:
        zip_path = os.path.join(os.path.expanduser('~'), 'Downloads', '18884178.zip')
        if not os.path.exists(zip_path):
            print(f"Climate-REF zip not found at {zip_path}")
            sys.exit(1)
        run_climate_ref_analysis(zip_path)

    elif '--csv' in args:
        idx = args.index('--csv')
        if idx + 1 >= len(args):
            print("Usage: --csv FILE.csv")
            sys.exit(1)
        filepath = args[idx + 1]
        state = encode_generic_csv(filepath)
        report = generate_weather_report(state, label=filepath)
        print(report["narrative"])

    elif '--synthetic' in args:
        run_demo()

    else:
        # Default: run everything we can
        run_demo()

        # Try climate data if available
        zip_path = os.path.join(os.path.expanduser('~'), 'Downloads', '18884178.zip')
        if os.path.exists(zip_path):
            print("\n\n" + "=" * 70)
            print("  BONUS: Real Climate-REF Data")
            print("=" * 70)
            try:
                run_climate_ref_analysis(zip_path)
            except Exception as e:
                print(f"Climate-REF analysis failed: {e}")
