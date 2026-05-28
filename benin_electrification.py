"""
Benin Least-Cost Electrification Analysis
==========================================


Approach:  Custom model informed by OnSSET methodology (Mentis et al. 2017)
           with one key modification: adding productive & institutional demand
           on top of OnSSET's residential-only estimation.

Planning horizon : 10 years (2024-2034)
Technologies     : Grid extension, Mini-grid (solar-hybrid), Solar Home Systems
Cost metric      : LCOE (USD/MWh) and cost per connection (USD)
Demand framework : World Bank Multi-Tier Framework (MTF), settlement-differentiated

References
----------
[1] Mentis et al. (2017) Env. Res. Lett. 12:085003  (OnSSET SSA study)
[2] Korkovelos et al. (2023) Nat. Comms. 14:4769     (OnSSET 40-country cost study)
[3] IRENA (2023) Renewable Power Generation Costs
[4] OnSSET docs: https://onsset.readthedocs.io
[5] World Bank Multi-Tier Framework: https://mtfrworkshop.esmap.org
"""

import json, logging, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS & ASSUMPTIONS
# ═══════════════════════════════════════════════════════════════════════

PLANNING_HORIZON = 10          # years  
BASE_YEAR        = 2024
TARGET_YEAR      = BASE_YEAR + PLANNING_HORIZON
DISCOUNT_RATE    = 0.10        # OnSSET standard for SSA [1]
HH_SIZE          = 4.5         # avg household size, West Africa (DHS)
POPULATION_GROWTH_RATE = 0.027 # Benin 2.7 %/yr (World Bank WDI)

# ── Demand tiers (kWh per household per year) ──────────────────────
# World Bank MTF [5]; values from OnSSET documentation [4]
#   Tier 1:   22 kWh/hh/yr  (basic lighting, phone charging)
#   Tier 2:   73 kWh/hh/yr  (lighting, TV, fan)
#   Tier 3:  365 kWh/hh/yr  (refrigerator, small productive use)
#   Tier 4:  730 kWh/hh/yr  (larger appliances)
#   Tier 5: 2190 kWh/hh/yr  (full modern supply)

DEMAND_TIERS = {
    "Tier1":   22,   # kWh/hh/yr
    "Tier2":   73,
    "Tier3":  365,
    "Tier4":  730,
    "Tier5": 2190,
}

# ── Productive & institutional demand multipliers ──────────────────
#  Productive demand (shops, mills, irrigation
#  pumps) were modelled as 20 % of residential and institutional demand 
#  (schools,health centres) as 10 % of residential, following common
#  electrification-planning practice (ESMAP 2019).
PRODUCTIVE_DEMAND_FACTOR    = 0.20   # 20 % of residential
INSTITUTIONAL_DEMAND_FACTOR = 0.10   # 10 % of residential

# ── Annual demand-per-connection growth ────────────────────────────
# As incomes rise, each household consumes more.
DEMAND_GROWTH_RATE = 0.05  # 5 % compound annual growth

# ── Technology cost assumptions ────────────────────────────────────
# Sources: OnSSET defaults [1][4], IRENA 2023 [3], SEforALL mini-grid
#          CAPEX/OPEX benchmark study (Aug 2024)

TECH_COSTS = {
    "grid": {
        "capex_per_km_usd":          7000,   # MV line $/km [1]
        "capex_per_connection_usd":   200,   # service drop + meter [1]
        "grid_generation_cost_usd_mwh": 80,  # avg grid tariff Benin [2]
        "opex_pct_capex":           0.03,    # O&M 3 % of CAPEX/yr [4]
        "max_viable_distance_km":     50,    # OnSSET threshold [1]
        "losses_pct":               0.15,    # T&D losses 15 % (SBEE Benin)
        "lifetime_years":             30,
    },
    "minigrid": {
        "capex_per_kw_usd":         3500,    # PV+battery+inverter [3]
        "capex_per_connection_usd":  400,    # distribution + metering [3]
        "opex_pct_capex":           0.05,    # O&M 5 % of CAPEX/yr [3]
        "battery_replacement_year":    7,    # Li-ion mid-life replacement
        "battery_cost_share":       0.40,    # battery is 40 % of system
        "capacity_factor":          0.18,    # solar PV in Benin (~4.4 PSH)
        "system_reserve_factor":    1.3,     # oversize for reliability
        "min_population":             50,    # smaller settlements can use SHS
        "max_population":          10000,    # too big → grid cheaper
        "lifetime_years":             20,
    },
    "shs": {
        "capex_tier1_usd":           150,    # ~15 Wp [3]
        "capex_tier2_usd":           400,    # 50 Wp SHS [3]
        "capex_tier3_usd":          1500,    # 250 Wp SHS needed for Tier 3 (365 kWh/yr)
        "opex_pct_capex":           0.02,    # 2 % O&M [4]
        "battery_replacement_year":    5,    # lead-acid or early Li-ion
        "battery_cost_share":       0.35,    # battery share of system
        "lifetime_years":             10,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# 1. DEMAND ESTIMATION
# ═══════════════════════════════════════════════════════════════════════

def assign_demand_tier(props: Dict) -> str:
    """
    Assign a demand tier to a settlement based on its characteristics.

    OnSSET uses a single tier for the whole country.  Here I differentiated
    settlements using population size, building density, proximity to grid,
    and presence of social infrastructure.

    Tier 1 → very small, remote, no services
    Tier 2 → small or remote
    Tier 3 → medium / near grid / has services
    Tier 4 → large / urban fringe
    Tier 5 → very large / urban centre
    """
    pop  = props.get("population", 0)
    dist = props.get("dist_to_existing_planned_transmission_lines_2017", 999)
    has_services = (props.get("has_health_facility", False) or
                    props.get("has_education_facility", False))
    nightlight   = props.get("has_nightlight", False)
    large_bldg   = props.get("Medium_and_large_buildings_pc", 0) or 0

    if pop >= 10000 or (pop >= 5000 and nightlight):
        return "Tier5"
    if pop >= 3000 or (pop >= 1000 and nightlight and has_services):
        return "Tier4"
    if pop >= 500 or (pop >= 200 and has_services) or (dist < 5):
        return "Tier3"
    if pop >= 100 or has_services or dist < 15:
        return "Tier2"
    return "Tier1"


def estimate_demand(props: Dict, cfg: Dict = None) -> pd.DataFrame:
    """
    Estimate annual electricity demand over the planning horizon.

    Returns a DataFrame with one row per year containing:
      year, population, num_hh, electrification_rate, num_connections,
      residential_mwh, productive_mwh, institutional_mwh, total_mwh,
      peak_kw, demand_per_conn_kwh_day

    Key formulas
    ------------
    Residential demand (year t):
        D_res(t) = connections(t) * tier_demand * (1+g)^t / 1000   [MWh]

    Productive demand:
        D_prod(t) = D_res(t) * 0.20

    Institutional demand:
        D_inst(t) = D_res(t) * 0.10

    Electrification rate growth:
        I used a linear ramp from the initial rate to 100 % over the
        planning horizon, consistent with SDG7 universal-access goals.
    """
    cfg = cfg or {}
    demand_growth = cfg.get("demand_growth_rate", DEMAND_GROWTH_RATE)

    pop0  = props.get("population", 100)
    dist  = props.get("dist_to_existing_planned_transmission_lines_2017", 999)
    tier  = assign_demand_tier(props)
    tier_kwh_hh_yr = DEMAND_TIERS[tier]

    # Initial electrification rate (proxy: closer to grid → higher)
    if dist < 2:
        init_elec = 0.60
    elif dist < 10:
        init_elec = 0.35
    elif dist < 30:
        init_elec = 0.15
    else:
        init_elec = 0.05

    rows = []
    for t in range(PLANNING_HORIZON + 1):
        year = BASE_YEAR + t

        # Population grows
        pop_t = pop0 * (1 + POPULATION_GROWTH_RATE) ** t
        num_hh = pop_t / HH_SIZE

        # Linear electrification ramp → 100 % by target year
        elec_rate = min(1.0, init_elec + (1.0 - init_elec) * (t / PLANNING_HORIZON))
        connections = num_hh * elec_rate

        # Per-connection demand grows with income / development
        demand_kwh_hh_yr = tier_kwh_hh_yr * (1 + demand_growth) ** t

        # Residential demand (MWh)
        res_mwh = connections * demand_kwh_hh_yr / 1000

        # Productive & institutional (modification to OnSSET)
        prod_mwh = res_mwh * PRODUCTIVE_DEMAND_FACTOR
        inst_mwh = res_mwh * INSTITUTIONAL_DEMAND_FACTOR
        total_mwh = res_mwh + prod_mwh + inst_mwh

        # Peak demand (for mini-grid sizing) – assume 25 % load factor
        peak_kw = (total_mwh * 1000) / (8760 * 0.25) if total_mwh > 0 else 0

        rows.append({
            "year": year,
            "population": pop_t,
            "num_hh": num_hh,
            "electrification_rate": elec_rate,
            "num_connections": connections,
            "tier": tier,
            "residential_mwh": res_mwh,
            "productive_mwh": prod_mwh,
            "institutional_mwh": inst_mwh,
            "total_mwh": total_mwh,
            "peak_kw": peak_kw,
            "demand_per_conn_kwh_day": demand_kwh_hh_yr / 365,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 2. TECHNOLOGY COST MODELS & LCOE
# ═══════════════════════════════════════════════════════════════════════

def pv(values: np.ndarray, rate: float) -> float:
    """
    Present value of a stream of annual cash-flows.

    PV = SUM( value(t) / (1 + rate)^t )   for t = 0 .. T

    """
    return sum(v / (1 + rate) ** i for i, v in enumerate(values))


def lcoe(costs: np.ndarray, energy_mwh: np.ndarray, rate: float) -> float:
    """
    Levelized Cost of Electricity (LCOE).

    LCOE = NPC(costs) / PV(energy delivered)   [USD/MWh]

    NPC (Net Present Cost) is the discounted sum of all project costs.
    Dividing by the discounted energy delivered gives the average cost
    per MWh over the project lifetime.
    """
    pv_e = pv(energy_mwh, rate)
    return pv(costs, rate) / pv_e if pv_e > 0 else np.inf


# ── 2a. Grid extension ────────────────────────────────────────────

def cost_grid(props: Dict, demand_df: pd.DataFrame, cfg: Dict = None) -> Dict:
    """
    Grid extension cost model.

    CAPEX = line_cost (distance * $/km) + connection_cost (connections * $/conn)
    OPEX  = CAPEX * opex_pct + energy purchase (generation cost * energy / (1-losses))
    NPC   = PV(all costs over lifetime)   [USD]
    LCOE  = NPC / PV(energy delivered)   [USD/MWh]
    """
    cfg = cfg or {}
    p   = dict(TECH_COSTS["grid"])
    p["capex_per_km_usd"] = cfg.get("grid_capex_per_km", p["capex_per_km_usd"])
    rate = cfg.get("discount_rate", DISCOUNT_RATE)
    dist_km = props.get("dist_to_existing_planned_transmission_lines_2017", 999)

    if dist_km > p["max_viable_distance_km"]:
        return {"viable": False, "reason": f"Distance {dist_km:.1f} km > {p['max_viable_distance_km']} km",
                "lcoe": np.inf, "cost_per_conn": np.inf}

    n_conn_final = demand_df.iloc[-1]["num_connections"]
    if n_conn_final < 1:
        return {"viable": False, "reason": "No connections", "lcoe": np.inf, "cost_per_conn": np.inf}

    capex_line = dist_km * p["capex_per_km_usd"]
    capex_conn = n_conn_final * p["capex_per_connection_usd"]
    capex      = capex_line + capex_conn

    costs, energy = [], []
    for i, row in demand_df.iterrows():
        e_delivered = row["total_mwh"]
        e_purchased = e_delivered / (1 - p["losses_pct"])  # account for losses
        opex = capex * p["opex_pct_capex"] * (1.02 ** i)   # 2 % inflation
        energy_cost = e_purchased * p["grid_generation_cost_usd_mwh"]

        if i == 0:
            costs.append(capex + opex + energy_cost)  # Year 0: build + run + buy energy
        else:
            costs.append(opex + energy_cost)   # Year 1-10: run + buy energy
        energy.append(e_delivered)

    lc = lcoe(np.array(costs), np.array(energy), rate)

    return {
        "viable": True,
        "lcoe": lc,
        "npc": pv(np.array(costs), rate),
        "cost_per_conn": capex / max(n_conn_final, 1),
        "capex": capex,
        "distance_km": dist_km,
    }


# ── 2b. Mini-grid (solar PV hybrid) ───────────────────────────────

def cost_minigrid(props: Dict, demand_df: pd.DataFrame, cfg: Dict = None) -> Dict:
    """
    Mini-grid cost model (solar PV + battery, no diesel).

    System sizing:
        annual_kWh  = avg(D_total) * 1000
        system_kW   = (annual_kWh / (8760 * capacity_factor)) * reserve_factor

    CAPEX = system_kW * $/kW + connections * $/conn
    OPEX  = CAPEX * opex_pct
    NPC   = PV(all costs including battery replacement)   [USD]
    LCOE  = NPC / PV(energy delivered)   [USD/MWh]
    """
    cfg = cfg or {}
    p   = dict(TECH_COSTS["minigrid"])
    p["capex_per_kw_usd"] = cfg.get("minigrid_capex_per_kw", p["capex_per_kw_usd"])
    rate = cfg.get("discount_rate", DISCOUNT_RATE)
    pop  = props.get("population", 0)

    if pop < p["min_population"]:
        return {"viable": False, "reason": f"Pop {pop} < {p['min_population']}",
                "lcoe": np.inf, "cost_per_conn": np.inf}
    if pop > p["max_population"]:
        return {"viable": False, "reason": f"Pop {pop} > {p['max_population']}",
                "lcoe": np.inf, "cost_per_conn": np.inf}

    avg_energy_mwh = demand_df["total_mwh"].mean()
    if avg_energy_mwh <= 0:
        return {"viable": False, "reason": "Zero demand", "lcoe": np.inf, "cost_per_conn": np.inf}

    # PV sizing: capacity needed to meet annual energy through solar generation
    #   annual_kwh = capacity_kw * 8760 * capacity_factor
    #   → capacity_kw = annual_kwh / (8760 * CF)
    # Then apply reserve factor for reliability margin.
    annual_kwh = avg_energy_mwh * 1000
    system_kw  = (annual_kwh / (8760 * p["capacity_factor"])) * p["system_reserve_factor"]
    n_conn_final = demand_df.iloc[-1]["num_connections"]

    capex_gen  = system_kw * p["capex_per_kw_usd"]
    capex_dist = n_conn_final * p["capex_per_connection_usd"]
    capex      = capex_gen + capex_dist

    costs, energy = [], []
    for i, row in demand_df.iterrows():
        opex = capex * p["opex_pct_capex"] * (1.02 ** i)
        replacement = 0
        if (BASE_YEAR + i) == BASE_YEAR + p["battery_replacement_year"]:
            replacement = capex_gen * p["battery_cost_share"] * 0.7  # cost decline

        if i == 0:
            costs.append(capex + opex)  # Year 0: build + run
        else:
            costs.append(opex + replacement)  # Year 1-10: run + battery replacement
        energy.append(row["total_mwh"])

    lc = lcoe(np.array(costs), np.array(energy), rate)

    return {
        "viable": True,
        "lcoe": lc,
        "npc": pv(np.array(costs), rate),
        "cost_per_conn": capex / max(n_conn_final, 1),
        "capex": capex,
        "system_kw": system_kw,
    }


# ── 2c. Solar home systems ─────────────────────────────────────────

def cost_shs(props: Dict, demand_df: pd.DataFrame, cfg: Dict = None) -> Dict:
    """
    Solar Home System cost model.

    System cost depends on demand tier:
        Tier 1-2 → pico / small SHS
        Tier 3   → mid-range SHS
        Tier 4-5 → not viable (demand exceeds SHS capacity)

    CAPEX = num_connections * (system_cost + logistics_cost)
    Logistics cost captures last-mile delivery to remote areas:
        $2/km * distance_to_grid (proxy for remoteness)
    OPEX  = 2 % of CAPEX/yr
    NPC   = PV(all costs including battery replacement)   [USD]
    LCOE  = NPC / PV(energy delivered)   [USD/MWh]
    """
    cfg  = cfg or {}
    mult = cfg.get("shs_cost_multiplier", 1.0)
    rate = cfg.get("discount_rate", DISCOUNT_RATE)
    p    = TECH_COSTS["shs"]
    tier = assign_demand_tier(props)

    if tier in ("Tier1", "Tier2"):
        unit_cost = p["capex_tier1_usd"] if tier == "Tier1" else p["capex_tier2_usd"]
    elif tier == "Tier3":
        unit_cost = p["capex_tier3_usd"]
    else:
        # Tier 4-5: demand exceeds what a single SHS can reliably supply
        # (730-2190 kWh/yr requires >500 Wp system, beyond typical SHS range)
        return {"viable": False, "reason": "Demand exceeds SHS capacity (Tier 4-5)",
                "lcoe": np.inf, "cost_per_conn": np.inf}

    unit_cost = unit_cost * mult   # apply sensitivity multiplier

    n_conn_final = demand_df.iloc[-1]["num_connections"]
    if n_conn_final < 1:
        return {"viable": False, "reason": "No connections", "lcoe": np.inf, "cost_per_conn": np.inf}

    # Last-mile logistics: delivering SHS kits to remote areas costs more.
    # $2/km per system is a simplified proxy (transport, warehousing, technician travel).
    dist_km = props.get("dist_to_existing_planned_transmission_lines_2017", 0)
    logistics_per_system = 2.0 * dist_km   # USD per system
    effective_unit_cost  = unit_cost + logistics_per_system

    capex = n_conn_final * effective_unit_cost

    costs, energy = [], []
    for i, row in demand_df.iterrows():
        opex = capex * p["opex_pct_capex"] * (1.02 ** i)
        replacement = 0
        if (BASE_YEAR + i) == BASE_YEAR + p["battery_replacement_year"]:
            replacement = capex * p["battery_cost_share"] * 0.8  # cost decline

        if i == 0:
            costs.append(capex + opex)  # Year 0: build + run
        else:
            costs.append(opex + replacement)  # Year 1-10: run + battery replacement
        energy.append(row["total_mwh"])

    lc = lcoe(np.array(costs), np.array(energy), rate)

    return {
        "viable": True,
        "lcoe": lc,
        "npc": pv(np.array(costs), rate),
        "cost_per_conn": unit_cost,
        "capex": capex,
        "unit_cost": unit_cost,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. LEAST-COST COMPARISON  &  PRIORITY SCORING
# ═══════════════════════════════════════════════════════════════════════

def analyze_settlement(props: Dict, cfg: Dict = None) -> Optional[Dict]:
    """Run demand + cost analysis for one settlement; return result dict."""
    pop = props.get("population", 0)
    if pop < 10:
        return None  # too small

    dist_km = props.get("dist_to_existing_planned_transmission_lines_2017", 999)
    demand_df = estimate_demand(props, cfg)

    grid_res = cost_grid(props, demand_df, cfg)
    mg_res   = cost_minigrid(props, demand_df, cfg)
    shs_res  = cost_shs(props, demand_df, cfg)

    # Least-cost selection
    options = {}
    if grid_res["viable"]:  options["Grid"]      = grid_res["lcoe"]
    if mg_res["viable"]:    options["Mini-grid"]  = mg_res["lcoe"]
    if shs_res["viable"]:   options["SHS"]        = shs_res["lcoe"]

    if not options:
        return None

    best_tech = min(options, key=options.get)
    best_lcoe = options[best_tech]

    # ── Priority score (for settlement prioritisation) ─────────────
    # Weighted composite:
    #   40 % population (larger → higher priority)
    #   25 % infrastructure (health/education → higher)
    #   20 % cost-effectiveness (lower LCOE → higher)
    #   15 % grid proximity (closer → easier)
    has_health = 1 if props.get("has_health_facility") else 0
    has_edu    = 1 if props.get("has_education_facility") else 0
    infra_score = (has_health * 0.6 + has_edu * 0.4)

    # Normalised 0-1 using reasonable ranges
    pop_norm   = min(pop / 10000, 1.0)
    lcoe_norm  = max(0, 1 - best_lcoe / 1000) if best_lcoe < np.inf else 0
    dist_norm  = max(0, 1 - dist_km / 100)

    priority = (0.40 * pop_norm +
                0.25 * infra_score +
                0.20 * lcoe_norm +
                0.15 * dist_norm)

    tier = assign_demand_tier(props)
    final_demand = demand_df.iloc[-1]["total_mwh"]
    final_conn   = demand_df.iloc[-1]["num_connections"]

    return {
        "settlement_id":        props.get("identifier"),
        "name":                 props.get("village_name", "Unknown"),
        "admin1":               props.get("admin_cgaz_1", ""),
        "admin2":               props.get("admin_cgaz_2", ""),
        "population":           pop,
        "num_buildings":        props.get("num_buildings", 0),
        "distance_to_grid_km":  round(dist_km, 2),
        "has_health":           has_health,
        "has_education":        has_edu,
        "has_nightlight":       1 if props.get("has_nightlight") else 0,
        "demand_tier":          tier,
        "final_connections":    round(final_conn, 1),
        "final_demand_mwh":     round(final_demand, 2),
        "grid_lcoe":            round(grid_res["lcoe"], 2) if grid_res["viable"] else None,
        "minigrid_lcoe":        round(mg_res["lcoe"], 2) if mg_res["viable"] else None,
        "shs_lcoe":             round(shs_res["lcoe"], 2) if shs_res["viable"] else None,
        "grid_cost_per_conn":   round(grid_res["cost_per_conn"], 2) if grid_res["viable"] else None,
        "minigrid_cost_per_conn": round(mg_res["cost_per_conn"], 2) if mg_res["viable"] else None,
        "shs_cost_per_conn":    round(shs_res["cost_per_conn"], 2) if shs_res["viable"] else None,
        "least_cost_technology": best_tech,
        "least_cost_lcoe":      round(best_lcoe, 2),
        "priority_score":       round(priority, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════

def run(data_path: str, output_dir: str, max_n: int = None,
        cfg: Dict = None, save: bool = True) -> pd.DataFrame:
    """
    Run full analysis.

    Parameters
    ----------
    data_path  : Path to GeoJSON file
    output_dir : Directory for CSV outputs
    max_n      : Cap number of settlements (None = all)
    cfg        : Parameter overrides for sensitivity analysis, e.g.:
                   {"discount_rate": 0.08, "grid_capex_per_km": 5500}
    save       : Write CSVs to disk (False for sensitivity runs)
    """
    cfg = cfg or {}

    if not cfg:  # only log for base-case runs
        log.info("=" * 70)
        log.info("BENIN LEAST-COST ELECTRIFICATION ANALYSIS")
        log.info("Approach: OnSSET-informed, with productive & institutional demand")
        log.info("=" * 70)
        log.info(f"Planning horizon : {BASE_YEAR}-{TARGET_YEAR} ({PLANNING_HORIZON} years)")
        log.info(f"Discount rate    : {cfg.get('discount_rate', DISCOUNT_RATE)*100:.0f} %")
        log.info(f"Demand growth    : {cfg.get('demand_growth_rate', DEMAND_GROWTH_RATE)*100:.0f} % per year")
        log.info(f"Pop. growth      : {POPULATION_GROWTH_RATE*100:.1f} % per year")

    with open(data_path) as f:
        geojson = json.load(f)
    total = len(geojson["features"])
    if not cfg:
        log.info(f"Settlements in file: {total}")

    results = []
    errors  = 0
    for i, feat in enumerate(geojson["features"]):
        if max_n and len(results) >= max_n:
            break
        try:
            r = analyze_settlement(feat["properties"], cfg)
            if r:
                results.append(r)
        except Exception as e:
            errors += 1

    df = pd.DataFrame(results)

    if save:
        out = Path(output_dir)
        df.to_csv(out / "benin_electrification_results.csv", index=False)
        top20 = df.nlargest(20, "priority_score")
        top20.to_csv(out / "top_20_priority_settlements.csv", index=False)

        log.info(f"Analyzed: {len(df)}  |  Skipped/errors: {errors}")
        log.info(f"Saved: benin_electrification_results.csv  ({len(df)} rows)")

        tech = df["least_cost_technology"].value_counts()
        log.info("\nTechnology distribution (least-cost):")
        for t, n in tech.items():
            pct = 100 * n / len(df)
            pop_share = df.loc[df["least_cost_technology"] == t, "population"].sum()
            log.info(f"  {t:12s}: {n:5d} settlements ({pct:5.1f} %)  | pop {pop_share:>10,}")

        log.info("\nDemand tier distribution:")
        for tier, cnt in df["demand_tier"].value_counts().sort_index().items():
            log.info(f"  {tier}: {cnt:5d} settlements ({100*cnt/len(df):.1f} %)")

        log.info("\nLCOE by technology (USD/MWh, where viable):")
        for col, label in [("grid_lcoe","Grid"),("minigrid_lcoe","Mini-grid"),("shs_lcoe","SHS")]:
            vals = df[col].dropna()
            if len(vals):
                log.info(f"  {label:12s}: median {vals.median():8.0f}  mean {vals.mean():8.0f}")

        log.info("\nCost per connection (USD, where viable):")
        for col, label in [("grid_cost_per_conn","Grid"),
                           ("minigrid_cost_per_conn","Mini-grid"),
                           ("shs_cost_per_conn","SHS")]:
            vals = df[col].dropna()
            if len(vals):
                log.info(f"  {label:12s}: median {vals.median():8.0f}  mean {vals.mean():8.0f}")

        top20 = df.nlargest(20, "priority_score")
        log.info("\nTop 10 priority settlements:")
        for _, r in top20.head(10).iterrows():
            log.info(f"  {r['name']:20s}  pop {r['population']:>7,}  "
                     f"{r['least_cost_technology']:10s}  "
                     f"LCOE ${r['least_cost_lcoe']:>7,.0f}/MWh  "
                     f"priority {r['priority_score']:.3f}")

        log.info("\n" + "=" * 70)
        log.info("ANALYSIS COMPLETE")
        log.info("=" * 70)

    return df


# ═══════════════════════════════════════════════════════════════════════
# 5. SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

# One-at-a-time (OAT) sensitivity: vary each parameter across 5 values
# while holding all others at their base-case value.
SENSITIVITY_PARAMS = {
    "discount_rate": {
        "label":  "Discount Rate",
        "unit":   "%",
        "values": [0.05, 0.08, 0.10, 0.12, 0.15],
        "labels": ["5 %", "8 %", "10 % (base)", "12 %", "15 %"],
        "base":   0.10,
    },
    "grid_capex_per_km": {
        "label":  "Grid Line Cost",
        "unit":   "$/km",
        "values": [4000, 5500, 7000, 9000, 11000],
        "labels": ["$4,000", "$5,500", "$7,000 (base)", "$9,000", "$11,000"],
        "base":   7000,
    },
    "shs_cost_multiplier": {
        "label":  "SHS System Cost",
        "unit":   "x base",
        "values": [0.70, 0.85, 1.00, 1.15, 1.30],
        "labels": ["-30 %", "-15 %", "Base", "+15 %", "+30 %"],
        "base":   1.00,
    },
    "demand_growth_rate": {
        "label":  "Demand Growth Rate",
        "unit":   "%/yr",
        "values": [0.02, 0.035, 0.05, 0.07, 0.09],
        "labels": ["2 %", "3.5 %", "5 % (base)", "7 %", "9 %"],
        "base":   0.05,
    },
    "minigrid_capex_per_kw": {
        "label":  "Mini-grid PV+Battery Cost",
        "unit":   "$/kW",
        "values": [2500, 3000, 3500, 4000, 4500],
        "labels": ["$2,500", "$3,000", "$3,500 (base)", "$4,000", "$4,500"],
        "base":   3500,
    },
}


def run_sensitivity(data_path: str, output_dir: str,
                    sample_n: int = 1000) -> pd.DataFrame:
    """
    One-at-a-time sensitivity analysis.

    For each parameter, run the model across 5 values (holding others at base).
    Record technology mix (% Grid / Mini-grid / SHS) and mean LCOE for each run.

    Parameters
    ----------
    data_path  : Path to GeoJSON file
    output_dir : Directory for CSV output
    sample_n   : Number of settlements to use per run (default 1000 for speed)

    Returns
    -------
    DataFrame with one row per (parameter, value) combination.
    """
    log.info("=" * 70)
    log.info("SENSITIVITY ANALYSIS — One-at-a-Time (OAT)")
    log.info(f"Parameters: {len(SENSITIVITY_PARAMS)}   "
             f"Values each: 5   "
             f"Settlements per run: {sample_n}")
    log.info("=" * 70)

    records = []

    for param_key, param_meta in SENSITIVITY_PARAMS.items():
        log.info(f"\nTesting: {param_meta['label']}")

        for val, val_label in zip(param_meta["values"], param_meta["labels"]):
            is_base = (val == param_meta["base"])
            cfg = {param_key: val}

            df = run(data_path, output_dir, max_n=sample_n, cfg=cfg, save=False)
            if len(df) == 0:
                continue

            n = len(df)
            tech = df["least_cost_technology"].value_counts()
            mean_lcoe = df["least_cost_lcoe"].mean()

            records.append({
                "parameter":      param_key,
                "param_label":    param_meta["label"],
                "value":          val,
                "value_label":    val_label,
                "is_base":        is_base,
                "n_settlements":  n,
                "grid_pct":       round(100 * tech.get("Grid",      0) / n, 1),
                "minigrid_pct":   round(100 * tech.get("Mini-grid", 0) / n, 1),
                "shs_pct":        round(100 * tech.get("SHS",       0) / n, 1),
                "mean_lcoe":      round(mean_lcoe, 1),
                "median_lcoe":    round(df["least_cost_lcoe"].median(), 1),
            })

            tag = " ← BASE" if is_base else ""
            log.info(f"  {val_label:20s}  Grid {records[-1]['grid_pct']:5.1f} %  "
                     f"MG {records[-1]['minigrid_pct']:4.1f} %  "
                     f"SHS {records[-1]['shs_pct']:5.1f} %  "
                     f"Mean LCOE ${mean_lcoe:,.0f}{tag}")

    sens_df = pd.DataFrame(records)
    out_path = Path(output_dir) / "sensitivity_results.csv"
    sens_df.to_csv(out_path, index=False)
    log.info(f"\nSaved: {out_path}")
    log.info("=" * 70)
    log.info("SENSITIVITY ANALYSIS COMPLETE")
    log.info("=" * 70)

    return sens_df


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    DATA   = "./data/Benin_settlement_properties.geojson"
    OUTPUT = "./outputs"

    # Run on full dataset
    df = run(DATA, OUTPUT)
