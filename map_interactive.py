"""
map_interactive.py
==================
Interactive GeoPandas + Folium map of Benin electrification analysis.

Features:
  - GeoPandas for proper GeoJSON loading and spatial operations
  - Interactive Folium map (Leaflet.js under the hood)
  - Live filter panel: technology, demand tier, population, distance,
    department, infrastructure flags
  - Toggleable layers: settlements, transmission lines, services
  - Rich click-to-inspect popups
  - Colour-coded by technology or demand tier
  - Legend and stats bar that update with filters
  - Multiple base maps (dark, satellite, topographic)

Output:
    outputs/map_03_interactive.html   (self-contained, works offline)

Installation (run once on your Mac):
    pip install geopandas folium pandas branca

Usage:
    python map_interactive.py
    open outputs/map_03_interactive.html
"""

import json
import csv
import math
import pandas as pd
import geopandas as gpd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
DATA_DIR   = Path("./data")
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SETTLEMENTS_FILE = DATA_DIR / "Benin_settlement_properties.geojson"
LINES_FILE       = DATA_DIR / "Benin_existing_transmission_lines_2017.geojson"
RESULTS_FILE     = OUTPUT_DIR / "benin_electrification_results.csv"

# ── Colours ────────────────────────────────────────────────────
TECH_COLORS = {
    "Grid":        "#22c55e",
    "Mini-grid":   "#f97316",
    "SHS":         "#38bdf8",
    "No analysis": "#475569",
}
TIER_COLORS = {
    "Tier1": "#fbbf24",
    "Tier2": "#f97316",
    "Tier3": "#ef4444",
    "Tier4": "#8b5cf6",
    "Tier5": "#1e40af",
}
VOLT_COLORS = {
    330: ("#fbbf24", 4),
    161: ("#f87171", 2.5),
    63:  ("#60a5fa", 1.5),
}

print("Loading data with GeoPandas...")

# ── Load with GeoPandas ────────────────────────────────────────
gdf = gpd.read_file(SETTLEMENTS_FILE)
lines_gdf = gpd.read_file(LINES_FILE)

print(f"  Settlements   : {len(gdf):,}  (CRS: {gdf.crs})")
print(f"  Lines         : {len(lines_gdf)}")

# ── Compute centroids ──────────────────────────────────────────
# GeoPandas computes accurate centroids from polygon geometry
gdf["centroid"] = gdf.geometry.centroid
gdf["lon"] = gdf["centroid"].x
gdf["lat"] = gdf["centroid"].y

# ── Load and merge analysis results ───────────────────────────
results = {}
if RESULTS_FILE.exists():
    with open(RESULTS_FILE) as f:
        for row in csv.DictReader(f):
            results[str(row["settlement_id"])] = {
                "tech":     row["least_cost_technology"],
                "lcoe":     float(row["least_cost_lcoe"]),
                "tier":     row["demand_tier"],
                "priority": float(row["priority_score"]),
                "grid_lcoe":     row.get("grid_lcoe", ""),
                "mg_lcoe":       row.get("minigrid_lcoe", ""),
                "shs_lcoe":      row.get("shs_lcoe", ""),
                "grid_cpc":      row.get("grid_cost_per_conn", ""),
                "mg_cpc":        row.get("minigrid_cost_per_conn", ""),
                "shs_cpc":       row.get("shs_cost_per_conn", ""),
                "connections":   row.get("final_connections", ""),
                "demand_mwh":    row.get("final_demand_mwh", ""),
            }
print(f"  Results loaded: {len(results):,}")

# ── Attach results to GeoDataFrame ────────────────────────────
gdf["sid"]      = gdf["identifier"].astype(str)
gdf["tech"]     = gdf["sid"].map(lambda x: results.get(x, {}).get("tech",     "No analysis"))
gdf["tier"]     = gdf["sid"].map(lambda x: results.get(x, {}).get("tier",     None))
gdf["lcoe"]     = gdf["sid"].map(lambda x: results.get(x, {}).get("lcoe",     None))
gdf["priority"] = gdf["sid"].map(lambda x: results.get(x, {}).get("priority", None))
gdf["analysed"] = gdf["tech"] != "No analysis"

# ── Build settlement data for JavaScript ──────────────────────
print("Building settlement records...")

records = []
for _, row in gdf.iterrows():
    sid = str(row.get("identifier", ""))
    res = results.get(sid, {})
    pop = row.get("population", 0) or 0
    dist = row.get("dist_to_existing_planned_transmission_lines_2017", 0) or 0

    # Marker radius (sqrt scale for area perception)
    radius = max(3, min(18, 3 + math.sqrt(pop / 200)))

    records.append({
        "id":       sid,
        "name":     str(row.get("village_name") or f"Settlement {sid}"),
        "admin1":   str(row.get("admin_cgaz_1") or ""),
        "admin2":   str(row.get("admin_cgaz_2") or ""),
        "lat":      round(row["lat"], 6),
        "lon":      round(row["lon"], 6),
        "pop":      int(pop),
        "buildings": int(row.get("num_buildings", 0) or 0),
        "dist":     round(float(dist), 2),
        "health":   bool(row.get("has_health_facility")),
        "edu":      bool(row.get("has_education_facility")),
        "night":    bool(row.get("has_nightlight")),
        "road":     bool(row.get("main_road_access")),
        "tech":     res.get("tech", "No analysis"),
        "tier":     res.get("tier", ""),
        "lcoe":     res.get("lcoe", ""),
        "priority": res.get("priority", ""),
        "grid_lcoe": res.get("grid_lcoe", ""),
        "mg_lcoe":   res.get("mg_lcoe", ""),
        "shs_lcoe":  res.get("shs_lcoe", ""),
        "grid_cpc":  res.get("grid_cpc", ""),
        "mg_cpc":    res.get("mg_cpc", ""),
        "shs_cpc":   res.get("shs_cpc", ""),
        "connections": res.get("connections", ""),
        "demand_mwh":  res.get("demand_mwh", ""),
        "radius":   round(radius, 1),
    })

# ── Build transmission line data for JavaScript ───────────────
line_records = []
for _, row in lines_gdf.iterrows():
    geom = row.geometry
    kv   = int(row.get("Voltage_KV", 161))
    col, wt = VOLT_COLORS.get(kv, ("#94a3b8", 1.5))

    # Extract coordinates from MultiLineString or LineString
    if geom.geom_type == "MultiLineString":
        for line in geom.geoms:
            coords = [[c[1], c[0]] for c in line.coords]
            line_records.append({
                "coords":  coords,
                "kv":      kv,
                "color":   col,
                "weight":  wt,
                "name":    str(row.get("Name", "")),
                "from_to": f"{row.get('From','')} → {row.get('To','')}",
                "km":      row.get("km", ""),
                "year":    row.get("Year", ""),
            })
    elif geom.geom_type == "LineString":
        coords = [[c[1], c[0]] for c in geom.coords]
        line_records.append({
            "coords":  coords,
            "kv":      kv,
            "color":   col,
            "weight":  wt,
            "name":    str(row.get("Name", "")),
            "from_to": f"{row.get('From','')} → {row.get('To','')}",
            "km":      row.get("km", ""),
            "year":    row.get("Year", ""),
        })

# Get unique departments for filter
departments = sorted(gdf["admin_cgaz_1"].dropna().unique().tolist())

print(f"  Settlement records : {len(records):,}")
print(f"  Line records       : {len(line_records)}")
print(f"  Departments        : {departments}")

# ── Serialise to JSON strings for embedding ────────────────────
import json as _json
records_json  = _json.dumps(records)
lines_json    = _json.dumps(line_records)
depts_json    = _json.dumps(departments)
tech_c_json   = _json.dumps(TECH_COLORS)
tier_c_json   = _json.dumps(TIER_COLORS)

# ═══════════════════════════════════════════════════════════════
# BUILD SELF-CONTAINED HTML
# ═══════════════════════════════════════════════════════════════

print("Building interactive HTML...")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Benin Electrification — Interactive Map</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    /* ── Reset & base ── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg:      #0a0f1e;
      --panel:   #0d1526;
      --raised:  #131f35;
      --border:  #1e3058;
      --border2: #253a6a;
      --text:    #e2e8f0;
      --muted:   #7b93b8;
      --accent:  #38bdf8;
      --green:   #22c55e;
      --orange:  #f97316;
      --blue:    #38bdf8;
      --slate:   #475569;
      --font-mono: 'IBM Plex Mono', monospace;
      --font-sans: 'IBM Plex Sans', sans-serif;
    }}
    html, body {{ height: 100%; overflow: hidden; background: var(--bg); }}
    body {{ display: flex; flex-direction: column; font-family: var(--font-sans); color: var(--text); }}

    /* ── Header ── */
    header {{
      flex-shrink: 0;
      height: 48px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      padding: 0 18px;
      gap: 16px;
      z-index: 2000;
    }}
    .hdr-title {{
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 600;
      letter-spacing: .12em;
      text-transform: uppercase;
      color: #fff;
    }}
    .hdr-meta {{
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--muted);
      letter-spacing: .06em;
    }}
    .hdr-spacer {{ flex: 1; }}
    .hdr-badge {{
      font-family: var(--font-mono);
      font-size: 9px;
      padding: 3px 8px;
      border-radius: 4px;
      background: rgba(56,189,248,.12);
      border: 1px solid rgba(56,189,248,.25);
      color: var(--accent);
      letter-spacing: .06em;
    }}

    /* ── Layout ── */
    .body {{ display: flex; flex: 1; overflow: hidden; }}
    #map {{ flex: 1; background: #060d1a; }}

    /* ── Sidebar ── */
    .sidebar {{
      width: 310px;
      flex-shrink: 0;
      background: var(--panel);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}

    /* ── Tab bar ── */
    .tabs {{
      display: flex;
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
    }}
    .tab {{
      flex: 1;
      padding: 11px 4px;
      font-family: var(--font-mono);
      font-size: 9.5px;
      letter-spacing: .1em;
      text-transform: uppercase;
      text-align: center;
      cursor: pointer;
      color: var(--muted);
      border-bottom: 2px solid transparent;
      transition: all .18s;
      user-select: none;
    }}
    .tab:hover {{ color: var(--text); }}
    .tab.active {{ color: #fff; border-bottom-color: var(--accent); }}

    /* ── Scrollable panel body ── */
    .panel-body {{
      flex: 1;
      overflow-y: auto;
      padding: 14px 14px 8px;
    }}
    .panel-body::-webkit-scrollbar {{ width: 4px; }}
    .panel-body::-webkit-scrollbar-track {{ background: transparent; }}
    .panel-body::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 2px; }}

    /* ── Section headings ── */
    .sec {{
      font-family: var(--font-mono);
      font-size: 8.5px;
      letter-spacing: .14em;
      text-transform: uppercase;
      color: var(--muted);
      margin: 16px 0 8px;
    }}
    .sec:first-child {{ margin-top: 0; }}

    /* ── Filter rows ── */
    .filter-row {{
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 5px 7px;
      border-radius: 6px;
      cursor: pointer;
      transition: background .14s;
      margin-bottom: 2px;
    }}
    .filter-row:hover {{ background: rgba(255,255,255,.04); }}
    .filter-row label {{
      font-size: 12px;
      color: var(--text);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 9px;
      flex: 1;
    }}
    input[type=checkbox] {{ width: 14px; height: 14px; accent-color: var(--accent); cursor: pointer; }}
    input[type=radio]    {{ accent-color: var(--accent); cursor: pointer; }}

    /* ── Swatch ── */
    .sw {{
      width: 11px; height: 11px;
      border-radius: 50%;
      flex-shrink: 0;
      border: 1.5px solid rgba(255,255,255,.15);
    }}
    .sw.sq {{ border-radius: 3px; }}
    .sw.ln {{ border-radius: 2px; height: 3px; width: 20px; }}

    .cnt {{
      font-family: var(--font-mono);
      font-size: 9px;
      color: var(--muted);
      margin-left: auto;
    }}

    /* ── Range slider ── */
    .range-group {{ margin-bottom: 12px; }}
    .range-label {{
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 5px;
    }}
    .range-val {{ color: var(--accent); font-family: var(--font-mono); font-size: 10px; }}
    input[type=range] {{
      width: 100%;
      height: 4px;
      accent-color: var(--accent);
      cursor: pointer;
      background: var(--raised);
      border-radius: 2px;
    }}

    /* ── Select dropdown ── */
    select {{
      width: 100%;
      background: var(--raised);
      border: 1px solid var(--border2);
      color: var(--text);
      padding: 6px 9px;
      border-radius: 6px;
      font-size: 11px;
      font-family: var(--font-sans);
      cursor: pointer;
      outline: none;
      margin-bottom: 10px;
    }}
    select:focus {{ border-color: var(--accent); }}

    /* ── Buttons ── */
    .btn {{
      width: 100%;
      padding: 8px;
      border-radius: 6px;
      font-family: var(--font-mono);
      font-size: 10px;
      letter-spacing: .1em;
      text-transform: uppercase;
      cursor: pointer;
      transition: all .15s;
    }}
    .btn-primary {{
      background: rgba(56,189,248,.14);
      border: 1px solid rgba(56,189,248,.3);
      color: var(--accent);
    }}
    .btn-primary:hover {{ background: rgba(56,189,248,.22); }}
    .btn-ghost {{
      background: transparent;
      border: 1px solid var(--border2);
      color: var(--muted);
      margin-top: 6px;
    }}
    .btn-ghost:hover {{ border-color: var(--text); color: var(--text); }}

    /* ── Stats footer ── */
    .stats-bar {{
      flex-shrink: 0;
      border-top: 1px solid var(--border);
      display: flex;
      padding: 10px 14px;
      gap: 8px;
    }}
    .stat {{ text-align: center; flex: 1; }}
    .stat-n {{
      font-family: var(--font-mono);
      font-size: 15px;
      font-weight: 600;
      color: #fff;
      line-height: 1;
    }}
    .stat-l {{
      font-size: 8.5px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-top: 3px;
      font-family: var(--font-mono);
    }}

    /* ── Tab panels ── */
    .tp {{ display: none; }}
    .tp.active {{ display: block; }}

    /* ── Legend swatch row ── */
    .leg-row {{
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 5px 7px;
      border-radius: 6px;
      margin-bottom: 2px;
      transition: background .14s;
      cursor: pointer;
    }}
    .leg-row:hover {{ background: rgba(255,255,255,.04); }}
    .leg-row.off {{ opacity: .3; }}
    .leg-lbl {{ font-size: 12px; color: var(--text); flex: 1; }}

    /* ── Leaflet popup ── */
    .leaflet-popup-content-wrapper {{
      background: transparent !important;
      box-shadow: none !important;
      padding: 0 !important;
      border-radius: 0 !important;
    }}
    .leaflet-popup-tip-container {{ display: none; }}
    .leaflet-popup-content {{ margin: 0 !important; }}

    .popup {{
      background: #0d1526ee;
      border: 1px solid #1e3058;
      border-radius: 10px;
      padding: 14px 16px;
      min-width: 240px;
      backdrop-filter: blur(14px);
      font-family: var(--font-sans);
    }}
    .popup-name {{
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 600;
      color: #fff;
      margin-bottom: 10px;
      line-height: 1.3;
    }}
    .popup-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 3px 0;
      border-bottom: 1px solid #1e3058;
      font-size: 11px;
    }}
    .popup-row:last-child {{ border: none; }}
    .pk {{ color: #7b93b8; }}
    .pv {{ color: #e2e8f0; font-weight: 500; }}
    .badge {{
      padding: 2px 8px;
      border-radius: 4px;
      font-family: var(--font-mono);
      font-size: 9.5px;
      font-weight: 600;
      letter-spacing: .05em;
    }}
    .popup-divider {{
      border: none;
      border-top: 1px solid #1e3058;
      margin: 7px 0;
    }}
  </style>
</head>
<body>

<!-- Header -->
<header>
  <div class="hdr-title">Benin Electrification</div>
  <div class="hdr-meta">17,205 settlements &nbsp;·&nbsp; 36 transmission lines</div>
  <div class="hdr-spacer"></div>
  <div class="hdr-badge" id="hdr-badge">Loading…</div>
</header>

<div class="body">
  <div id="map"></div>

  <div class="sidebar">
    <div class="tabs">
      <div class="tab active"  onclick="switchTab('filter')">Filters</div>
      <div class="tab"         onclick="switchTab('layers')">Layers</div>
      <div class="tab"         onclick="switchTab('legend')">Legend</div>
    </div>

    <div class="panel-body">

      <!-- ═══ FILTERS TAB ═══ -->
      <div class="tp active" id="tp-filter">

        <div class="sec">Technology</div>
        <div id="tech-checks"></div>

        <div class="sec">Demand Tier (MTF)</div>
        <div id="tier-checks"></div>

        <div class="sec">Department</div>
        <select id="sel-dept" onchange="applyFilters()">
          <option value="">All departments</option>
        </select>

        <div class="sec">Population range</div>
        <div class="range-group">
          <div class="range-label">
            <span>Min population</span>
            <span class="range-val" id="lbl-pop">0</span>
          </div>
          <input type="range" id="rng-pop" min="0" max="50000" value="0" step="100"
                 oninput="document.getElementById('lbl-pop').textContent=Number(this.value).toLocaleString(); applyFilters()">
        </div>

        <div class="sec">Distance to grid</div>
        <div class="range-group">
          <div class="range-label">
            <span>Max distance (km)</span>
            <span class="range-val" id="lbl-dist">111</span>
          </div>
          <input type="range" id="rng-dist" min="0" max="111" value="111" step="1"
                 oninput="document.getElementById('lbl-dist').textContent=this.value; applyFilters()">
        </div>

        <div class="sec">LCOE ceiling (analysed only)</div>
        <div class="range-group">
          <div class="range-label">
            <span>Max LCOE (USD/MWh)</span>
            <span class="range-val" id="lbl-lcoe">2000</span>
          </div>
          <input type="range" id="rng-lcoe" min="0" max="2000" value="2000" step="50"
                 oninput="document.getElementById('lbl-lcoe').textContent=this.value; applyFilters()">
        </div>

        <div class="sec">Must have</div>
        <div class="filter-row"><label><input type="checkbox" id="chk-health" onchange="applyFilters()"> Health facility</label></div>
        <div class="filter-row"><label><input type="checkbox" id="chk-edu"    onchange="applyFilters()"> Education facility</label></div>
        <div class="filter-row"><label><input type="checkbox" id="chk-road"   onchange="applyFilters()"> Main road access</label></div>
        <div class="filter-row"><label><input type="checkbox" id="chk-night"  onchange="applyFilters()"> Nightlight detected</label></div>
        <div class="filter-row"><label><input type="checkbox" id="chk-analysed" onchange="applyFilters()"> Analysed only</label></div>

        <div style="margin-top:14px; display:flex; flex-direction:column; gap:6px">
          <button class="btn btn-primary" onclick="resetFilters()">Reset all filters</button>
          <button class="btn btn-ghost"   onclick="map.fitBounds([[5.9,0.55],[12.7,4.0]])">Reset view</button>
        </div>
        <div style="margin-top:10px; font-family:var(--font-mono); font-size:9.5px; color:var(--muted); text-align:center" id="filter-status"></div>
      </div>

      <!-- ═══ LAYERS TAB ═══ -->
      <div class="tp" id="tp-layers">
        <div class="sec">Base map</div>
        <div class="filter-row"><label><input type="radio" name="base" value="dark"  checked onchange="setBase(this.value)"> Dark (default)</label></div>
        <div class="filter-row"><label><input type="radio" name="base" value="sat"         onchange="setBase(this.value)"> Satellite</label></div>
        <div class="filter-row"><label><input type="radio" name="base" value="topo"        onchange="setBase(this.value)"> Topographic</label></div>

        <div class="sec">Colour settlements by</div>
        <div class="filter-row"><label><input type="radio" name="colmode" value="tech" checked onchange="setColorMode(this.value)"> Technology</label></div>
        <div class="filter-row"><label><input type="radio" name="colmode" value="tier"       onchange="setColorMode(this.value)"> Demand tier</label></div>
        <div class="filter-row"><label><input type="radio" name="colmode" value="dist"       onchange="setColorMode(this.value)"> Distance to grid</label></div>
        <div class="filter-row"><label><input type="radio" name="colmode" value="pop"        onchange="setColorMode(this.value)"> Population</label></div>

        <div class="sec">Overlay layers</div>
        <div class="filter-row"><label><input type="checkbox" id="lyr-settlements" checked onchange="toggleLayer('settlements',this.checked)"> All settlements</label></div>
        <div class="filter-row"><label><input type="checkbox" id="lyr-lines"       checked onchange="toggleLayer('lines',this.checked)"> Transmission lines</label></div>
        <div class="filter-row"><label><input type="checkbox" id="lyr-health"             onchange="toggleLayer('health',this.checked)"> Health facilities</label></div>
        <div class="filter-row"><label><input type="checkbox" id="lyr-edu"                onchange="toggleLayer('edu',this.checked)"> Education facilities</label></div>
        <div class="filter-row"><label><input type="checkbox" id="lyr-night"              onchange="toggleLayer('night',this.checked)"> Nightlight detected</label></div>
        <div class="filter-row"><label><input type="checkbox" id="lyr-road"               onchange="toggleLayer('road',this.checked)"> Main road access</label></div>

        <div class="sec">Marker size</div>
        <div class="range-group">
          <div class="range-label"><span>Scale factor</span><span class="range-val" id="lbl-size">1.0</span></div>
          <input type="range" id="rng-size" min="0.3" max="3" value="1" step="0.1"
                 oninput="document.getElementById('lbl-size').textContent=parseFloat(this.value).toFixed(1); setSizeScale(this.value)">
        </div>
      </div>

      <!-- ═══ LEGEND TAB ═══ -->
      <div class="tp" id="tp-legend">

        <div class="sec">Technology (click to toggle)</div>
        <div id="legend-tech"></div>

        <div class="sec">Demand tier (World Bank MTF)</div>
        <div id="legend-tier"></div>

        <div class="sec">Transmission lines</div>
        <div class="leg-row"><div class="sw ln" style="background:#fbbf24"></div><span class="leg-lbl">330 kV — National backbone</span><span class="cnt">1</span></div>
        <div class="leg-row"><div class="sw ln" style="background:#f87171"></div><span class="leg-lbl">161 kV — High voltage</span><span class="cnt">24</span></div>
        <div class="leg-row"><div class="sw ln" style="background:#60a5fa"></div><span class="leg-lbl">63 kV  — Medium voltage</span><span class="cnt">11</span></div>

        <div class="sec">Services</div>
        <div class="leg-row"><div class="sw" style="background:none;border:2px solid #22c55e;border-radius:50%"></div><span class="leg-lbl">Health facility</span></div>
        <div class="leg-row"><div class="sw sq" style="background:none;border:2px solid #f59e0b"></div><span class="leg-lbl">Education facility</span></div>
        <div class="leg-row"><div class="sw" style="background:#fef08a"></div><span class="leg-lbl">Nightlight detected</span></div>
        <div class="leg-row"><div class="sw sq" style="background:#60a5fa99"></div><span class="leg-lbl">Main road access</span></div>

        <div class="sec">Settlement size (radius)</div>
        <div style="display:flex;align-items:center;gap:14px;padding:6px 8px;flex-wrap:wrap">
          <div style="display:flex;align-items:center;gap:6px"><svg width="12" height="12"><circle cx="6" cy="6" r="3" fill="#94a3b8"/></svg><span style="font-size:11px;color:var(--muted)">&lt;100</span></div>
          <div style="display:flex;align-items:center;gap:6px"><svg width="16" height="16"><circle cx="8" cy="8" r="5" fill="#94a3b8"/></svg><span style="font-size:11px;color:var(--muted)">2,000</span></div>
          <div style="display:flex;align-items:center;gap:6px"><svg width="22" height="22"><circle cx="11" cy="11" r="9" fill="#94a3b8"/></svg><span style="font-size:11px;color:var(--muted)">15,000+</span></div>
        </div>
      </div>
    </div>

    <!-- Stats bar -->
    <div class="stats-bar">
      <div class="stat"><div class="stat-n" id="st-vis">—</div><div class="stat-l">Visible</div></div>
      <div class="stat"><div class="stat-n" id="st-an">—</div><div class="stat-l">Analysed</div></div>
      <div class="stat"><div class="stat-n" id="st-pop">—</div><div class="stat-l">Pop (k)</div></div>
      <div class="stat"><div class="stat-n" id="st-lcoe">—</div><div class="stat-l">Med. LCOE</div></div>
    </div>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════
// DATA (embedded by Python)
// ═══════════════════════════════════════════════════════
const SETTLEMENTS = {records_json};
const LINES       = {lines_json};
const DEPARTMENTS = {depts_json};
const TECH_COLORS = {tech_c_json};
const TIER_COLORS = {tier_c_json};

const DIST_COLORS = [
  [0,   2,   "#22c55e"],
  [2,   10,  "#84cc16"],
  [10,  25,  "#facc15"],
  [25,  50,  "#f97316"],
  [50,  100, "#ef4444"],
  [100, 999, "#7f1d1d"],
];
const POP_COLORS = [
  [0,     100,   "#94a3b8"],
  [100,   500,   "#86efac"],
  [500,   2000,  "#facc15"],
  [2000,  5000,  "#f97316"],
  [5000,  15000, "#dc2626"],
  [15000, 1e9,   "#7f1d1d"],
];

// ═══════════════════════════════════════════════════════
// MAP INIT
// ═══════════════════════════════════════════════════════
const map = L.map('map', {{center:[9.3,2.3], zoom:7, zoomControl:true}});

const baseTiles = {{
  dark: L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{attribution:'&copy; CartoDB',maxZoom:18}}),
  sat:  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{attribution:'&copy; Esri',maxZoom:18}}),
  topo: L.tileLayer('https://{{s}}.tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png', {{attribution:'&copy; OpenTopoMap',maxZoom:17}}),
}};
baseTiles.dark.addTo(map);
let currentBase = 'dark';

// ═══════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════
let colorMode   = 'tech';
let sizeScale   = 1.0;
let hiddenTechs = new Set();
let hiddenTiers = new Set();

// Filter state
let fPopMin  = 0, fDistMax = 111, fLcoeMax = 2000;
let fHealth  = false, fEdu = false, fRoad = false, fNight = false, fAnalysed = false;
let fDept    = '';

// Layer groups
const layerGroups = {{
  settlements: L.layerGroup().addTo(map),
  lines:       L.layerGroup().addTo(map),
  health:      L.layerGroup(),
  edu:         L.layerGroup(),
  night:       L.layerGroup(),
  road:        L.layerGroup(),
}};

// ═══════════════════════════════════════════════════════
// COLOUR HELPERS
// ═══════════════════════════════════════════════════════
function getColor(s) {{
  if (colorMode === 'tech') return TECH_COLORS[s.tech] || '#475569';
  if (colorMode === 'tier') return TIER_COLORS[s.tier] || '#475569';
  if (colorMode === 'dist') {{
    for (const [lo,hi,c] of DIST_COLORS) if (s.dist >= lo && s.dist < hi) return c;
    return '#7f1d1d';
  }}
  if (colorMode === 'pop') {{
    for (const [lo,hi,c] of POP_COLORS) if (s.pop >= lo && s.pop < hi) return c;
    return '#7f1d1d';
  }}
  return '#475569';
}}

// ═══════════════════════════════════════════════════════
// FILTER LOGIC
// ═══════════════════════════════════════════════════════
function passesFilter(s) {{
  if (hiddenTechs.has(s.tech))           return false;
  if (s.tier && hiddenTiers.has(s.tier)) return false;
  if (s.pop < fPopMin)                   return false;
  if (s.dist > fDistMax)                 return false;
  if (s.lcoe !== '' && fLcoeMax < 2000 && parseFloat(s.lcoe) > fLcoeMax) return false;
  if (fHealth   && !s.health)            return false;
  if (fEdu      && !s.edu)               return false;
  if (fRoad     && !s.road)              return false;
  if (fNight    && !s.night)             return false;
  if (fAnalysed && s.tech==='No analysis') return false;
  if (fDept     && s.admin1 !== fDept)   return false;
  return true;
}}

// ═══════════════════════════════════════════════════════
// POPUP BUILDER
// ═══════════════════════════════════════════════════════
function makePopup(s) {{
  const tc = TECH_COLORS[s.tech] || '#475569';
  const badge = `<span class="badge" style="background:${{tc}}22;color:${{tc}};border:1px solid ${{tc}}44">${{s.tech}}</span>`;

  const fmtLcoe = v => v ? '$'+Number(v).toLocaleString(undefined,{{maximumFractionDigits:0}})+'/MWh' : '—';
  const fmtCpc  = v => v ? '$'+Number(v).toLocaleString(undefined,{{maximumFractionDigits:0}}) : '—';

  let analysisRows = '';
  if (s.tech !== 'No analysis') {{
    analysisRows = `
      <hr class="popup-divider">
      <div class="popup-row"><span class="pk">Technology</span><span class="pv">${{badge}}</span></div>
      <div class="popup-row"><span class="pk">Demand tier</span><span class="pv">${{s.tier || '—'}}</span></div>
      <div class="popup-row"><span class="pk">LCOE</span><span class="pv">${{fmtLcoe(s.lcoe)}}</span></div>
      <div class="popup-row"><span class="pk">Priority score</span><span class="pv">${{s.priority ? Number(s.priority).toFixed(3) : '—'}}</span></div>
      <div class="popup-row"><span class="pk">Final connections</span><span class="pv">${{s.connections ? Number(s.connections).toLocaleString(undefined,{{maximumFractionDigits:0}}) : '—'}}</span></div>
      <div class="popup-row"><span class="pk">Demand 2034</span><span class="pv">${{s.demand_mwh ? Number(s.demand_mwh).toLocaleString(undefined,{{maximumFractionDigits:1}})+' MWh' : '—'}}</span></div>
      <hr class="popup-divider">
      <div class="popup-row"><span class="pk">Grid LCOE</span><span class="pv">${{fmtLcoe(s.grid_lcoe)}}</span></div>
      <div class="popup-row"><span class="pk">Mini-grid LCOE</span><span class="pv">${{fmtLcoe(s.mg_lcoe)}}</span></div>
      <div class="popup-row"><span class="pk">SHS LCOE</span><span class="pv">${{fmtLcoe(s.shs_lcoe)}}</span></div>
      <div class="popup-row"><span class="pk">Grid cost/conn</span><span class="pv">${{fmtCpc(s.grid_cpc)}}</span></div>
      <div class="popup-row"><span class="pk">Mini-grid cost/conn</span><span class="pv">${{fmtCpc(s.mg_cpc)}}</span></div>
      <div class="popup-row"><span class="pk">SHS cost/conn</span><span class="pv">${{fmtCpc(s.shs_cpc)}}</span></div>
    `;
  }}

  const svcs = [s.health?'Health':'', s.edu?'Education':'', s.night?'Nightlight':'', s.road?'Road':'']
    .filter(Boolean).join(', ') || 'None';

  return `<div class="popup">
    <div class="popup-name">${{s.name}}</div>
    <div class="popup-row"><span class="pk">Department</span><span class="pv">${{s.admin1}}</span></div>
    <div class="popup-row"><span class="pk">Commune</span><span class="pv">${{s.admin2}}</span></div>
    <div class="popup-row"><span class="pk">Population</span><span class="pv">${{s.pop.toLocaleString()}}</span></div>
    <div class="popup-row"><span class="pk">Buildings</span><span class="pv">${{s.buildings.toLocaleString()}}</span></div>
    <div class="popup-row"><span class="pk">Dist. to grid</span><span class="pv">${{s.dist}} km</span></div>
    <div class="popup-row"><span class="pk">Services</span><span class="pv">${{svcs}}</span></div>
    ${{analysisRows}}
  </div>`;
}}

// ═══════════════════════════════════════════════════════
// BUILD SETTLEMENT MARKERS
// ═══════════════════════════════════════════════════════
// Store marker references for fast recolouring
const markerMap = new Map();   // sid → circle marker

function buildSettlements() {{
  layerGroups.settlements.clearLayers();
  layerGroups.health.clearLayers();
  layerGroups.edu.clearLayers();
  layerGroups.night.clearLayers();
  layerGroups.road.clearLayers();

  let nVis=0, nAn=0, totalPop=0;
  const lcoeVals = [];
  const techCnts = Object.fromEntries(Object.keys(TECH_COLORS).map(t=>[t,0]));

  for (const s of SETTLEMENTS) {{
    const visible = passesFilter(s);
    const col = getColor(s);
    const r   = s.radius * sizeScale;

    let marker = markerMap.get(s.id);
    if (!marker) {{
      marker = L.circleMarker([s.lat, s.lon], {{
        radius:      r,
        fillColor:   col,
        color:       'rgba(255,255,255,0.15)',
        weight:      0.5,
        opacity:     1,
        fillOpacity: 0.82,
      }});
      marker.bindPopup(makePopup(s), {{maxWidth:280, className:''}});
      markerMap.set(s.id, marker);
    }} else {{
      marker.setStyle({{
        radius:    r,
        fillColor: col,
      }});
    }}

    if (visible) {{
      layerGroups.settlements.addLayer(marker);
      nVis++;
      if (s.tech !== 'No analysis') {{ nAn++; if(s.lcoe) lcoeVals.push(parseFloat(s.lcoe)); }}
      totalPop += s.pop;
      techCnts[s.tech] = (techCnts[s.tech]||0) + 1;

      // Service overlay markers
      if (s.health) {{
        const m = L.circleMarker([s.lat,s.lon],{{radius:5,fillColor:'#fff',color:'#22c55e',weight:2,fillOpacity:.9}});
        m.bindPopup(makePopup(s),{{maxWidth:280}});
        layerGroups.health.addLayer(m);
      }}
      if (s.edu) {{
        const m = L.circleMarker([s.lat,s.lon],{{radius:5,fillColor:'#fff',color:'#f59e0b',weight:2,fillOpacity:.9}});
        m.bindPopup(makePopup(s),{{maxWidth:280}});
        layerGroups.edu.addLayer(m);
      }}
      if (s.night) {{
        const m = L.circleMarker([s.lat,s.lon],{{radius:4,fillColor:'#fef08a',color:'#ca8a04',weight:1,fillOpacity:.8}});
        m.bindPopup(makePopup(s),{{maxWidth:280}});
        layerGroups.night.addLayer(m);
      }}
      if (s.road) {{
        const m = L.circleMarker([s.lat,s.lon],{{radius:3,fillColor:'#60a5fa',color:'#60a5fa',weight:1,fillOpacity:.5}});
        m.bindPopup(makePopup(s),{{maxWidth:280}});
        layerGroups.road.addLayer(m);
      }}
    }}
  }}

  // Stats bar
  const medLcoe = lcoeVals.length
    ? lcoeVals.sort((a,b)=>a-b)[Math.floor(lcoeVals.length/2)]
    : null;

  document.getElementById('st-vis').textContent  = nVis.toLocaleString();
  document.getElementById('st-an').textContent   = nAn.toLocaleString();
  document.getElementById('st-pop').textContent  = Math.round(totalPop/1000).toLocaleString();
  document.getElementById('st-lcoe').textContent = medLcoe ? '$'+Math.round(medLcoe) : '—';

  const total = nVis || 1;
  document.getElementById('hdr-badge').textContent =
    `${{nVis.toLocaleString()}} settlements visible`;

  document.getElementById('filter-status').textContent =
    `${{nVis.toLocaleString()}} settlements · ${{nAn.toLocaleString()}} analysed`;

  // Update legend counts
  for (const [tech, col] of Object.entries(TECH_COLORS)) {{
    const el = document.getElementById('lc-'+tech.replace(/[^a-z]/gi,'_'));
    if (el) el.textContent = (techCnts[tech]||0).toLocaleString();
  }}
}}

// ═══════════════════════════════════════════════════════
// BUILD TRANSMISSION LINES (static — not filtered)
// ═══════════════════════════════════════════════════════
function buildLines() {{
  layerGroups.lines.clearLayers();
  for (const l of LINES) {{
    const poly = L.polyline(l.coords, {{
      color:   l.color,
      weight:  l.weight,
      opacity: 0.88,
    }});
    poly.bindPopup(`<div class="popup">
      <div class="popup-name">${{l.name}}</div>
      <div class="popup-row"><span class="pk">Route</span><span class="pv">${{l.from_to}}</span></div>
      <div class="popup-row"><span class="pk">Voltage</span><span class="pv">${{l.kv}} kV</span></div>
      <div class="popup-row"><span class="pk">Length</span><span class="pv">${{l.km}} km</span></div>
      <div class="popup-row"><span class="pk">Year</span><span class="pv">${{l.year}}</span></div>
    </div>`, {{maxWidth:260}});
    layerGroups.lines.addLayer(poly);
  }}
}}

// ═══════════════════════════════════════════════════════
// FILTER CONTROLS
// ═══════════════════════════════════════════════════════
function applyFilters() {{
  fPopMin    = parseInt(document.getElementById('rng-pop').value);
  fDistMax   = parseInt(document.getElementById('rng-dist').value);
  fLcoeMax   = parseInt(document.getElementById('rng-lcoe').value);
  fHealth    = document.getElementById('chk-health').checked;
  fEdu       = document.getElementById('chk-edu').checked;
  fRoad      = document.getElementById('chk-road').checked;
  fNight     = document.getElementById('chk-night').checked;
  fAnalysed  = document.getElementById('chk-analysed').checked;
  fDept      = document.getElementById('sel-dept').value;
  buildSettlements();
}}

function resetFilters() {{
  document.getElementById('rng-pop').value  = 0;
  document.getElementById('rng-dist').value = 111;
  document.getElementById('rng-lcoe').value = 2000;
  document.getElementById('lbl-pop').textContent  = '0';
  document.getElementById('lbl-dist').textContent = '111';
  document.getElementById('lbl-lcoe').textContent = '2000';
  ['chk-health','chk-edu','chk-road','chk-night','chk-analysed']
    .forEach(id => document.getElementById(id).checked = false);
  document.getElementById('sel-dept').value = '';
  fPopMin=0; fDistMax=111; fLcoeMax=2000;
  fHealth=fEdu=fRoad=fNight=fAnalysed=false; fDept='';
  hiddenTechs.clear(); hiddenTiers.clear();
  document.querySelectorAll('.leg-row').forEach(el => el.classList.remove('off'));
  buildSettlements();
}}

function toggleLayer(name, on) {{
  if (on) map.addLayer(layerGroups[name]);
  else    map.removeLayer(layerGroups[name]);
}}

function setBase(val) {{
  map.removeLayer(baseTiles[currentBase]);
  baseTiles[val].addTo(map);
  currentBase = val;
}}

function setColorMode(val) {{
  colorMode = val;
  buildSettlements();
}}

function setSizeScale(val) {{
  sizeScale = parseFloat(val);
  buildSettlements();
}}

function switchTab(name) {{
  document.querySelectorAll('.tab').forEach((t,i) => {{
    const names = ['filter','layers','legend'];
    t.classList.toggle('active', names[i]===name);
  }});
  document.querySelectorAll('.tp').forEach(p => {{
    p.classList.toggle('active', p.id==='tp-'+name);
  }});
}}

// ═══════════════════════════════════════════════════════
// BUILD DYNAMIC UI ELEMENTS
// ═══════════════════════════════════════════════════════
function buildFilterChecks() {{
  // Technology checkboxes
  const tc = document.getElementById('tech-checks');
  for (const [tech,col] of Object.entries(TECH_COLORS)) {{
    const d = document.createElement('div');
    d.className = 'filter-row';
    d.innerHTML = `<label>
      <input type="checkbox" id="tech-${{tech.replace(/[^a-z]/gi,'_')}}" checked
             onchange="toggleFilterTech('${{tech}}',this.checked)">
      <span class="sw" style="background:${{col}}"></span>
      ${{tech}}
    </label>`;
    tc.appendChild(d);
  }}

  // Tier checkboxes
  const tierLabels = {{
    'Tier1':'Tier 1 — Basic (22 kWh/yr)',
    'Tier2':'Tier 2 — Low (73 kWh/yr)',
    'Tier3':'Tier 3 — Medium (365 kWh/yr)',
    'Tier4':'Tier 4 — High (730 kWh/yr)',
    'Tier5':'Tier 5 — Full (2,190 kWh/yr)',
  }};
  const tirc = document.getElementById('tier-checks');
  for (const [tier,col] of Object.entries(TIER_COLORS)) {{
    const d = document.createElement('div');
    d.className = 'filter-row';
    d.innerHTML = `<label>
      <input type="checkbox" id="tier-${{tier}}" checked
             onchange="toggleFilterTier('${{tier}}',this.checked)">
      <span class="sw sq" style="background:${{col}}"></span>
      ${{tierLabels[tier]||tier}}
    </label>`;
    tirc.appendChild(d);
  }}

  // Dept select
  const sel = document.getElementById('sel-dept');
  DEPARTMENTS.forEach(d => {{
    const o = document.createElement('option');
    o.value = d; o.textContent = d;
    sel.appendChild(o);
  }});

  // Legend tech
  const lt = document.getElementById('legend-tech');
  for (const [tech,col] of Object.entries(TECH_COLORS)) {{
    const d = document.createElement('div');
    d.className = 'leg-row';
    d.onclick = () => toggleLegendTech(tech, d);
    d.innerHTML = `<div class="sw" style="background:${{col}}"></div>
      <span class="leg-lbl">${{tech}}</span>
      <span class="cnt" id="lc-${{tech.replace(/[^a-z]/gi,'_')}}">—</span>`;
    lt.appendChild(d);
  }}

  // Legend tier
  const lr = document.getElementById('legend-tier');
  const tierDesc = {{
    Tier1:'Basic — 22 kWh/yr',Tier2:'Low — 73 kWh/yr',
    Tier3:'Medium — 365 kWh/yr',Tier4:'High — 730 kWh/yr',
    Tier5:'Full — 2,190 kWh/yr'
  }};
  for (const [tier,col] of Object.entries(TIER_COLORS)) {{
    const d = document.createElement('div');
    d.className = 'leg-row';
    d.innerHTML = `<div class="sw sq" style="background:${{col}}"></div>
      <span class="leg-lbl">${{tier}} — ${{tierDesc[tier]||''}}</span>`;
    lr.appendChild(d);
  }}
}}

function toggleFilterTech(tech, on) {{
  if (on) hiddenTechs.delete(tech); else hiddenTechs.add(tech);
  buildSettlements();
}}
function toggleFilterTier(tier, on) {{
  if (on) hiddenTiers.delete(tier); else hiddenTiers.add(tier);
  buildSettlements();
}}
function toggleLegendTech(tech, el) {{
  if (hiddenTechs.has(tech)) {{
    hiddenTechs.delete(tech);
    el.classList.remove('off');
    const chk = document.getElementById('tech-'+tech.replace(/[^a-z]/gi,'_'));
    if(chk) chk.checked = true;
  }} else {{
    hiddenTechs.add(tech);
    el.classList.add('off');
    const chk = document.getElementById('tech-'+tech.replace(/[^a-z]/gi,'_'));
    if(chk) chk.checked = false;
  }}
  buildSettlements();
}}

// ═══════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════
buildFilterChecks();
buildLines();
buildSettlements();
</script>
</body>
</html>"""

# ── Write output ───────────────────────────────────────
out = OUTPUT_DIR / "map_03_interactive.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = out.stat().st_size / 1024
print(f"\nSaved: {out}  ({size_kb:.0f} KB)")
print(f"Open with:  open {out}")
print("\nInstall dependencies if needed:")
print("  pip install geopandas folium pandas")
