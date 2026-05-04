"""
cctv2_pipeline.py — Pure Python replacement for cctv2.py (QGIS Processing Algorithm)

Produces identical outputs using shapely, geopandas, scipy, numpy.
No QGIS/PyQGIS dependency.

Inputs (all file paths):
    building:      GeoJSON polygon file
    parking_area:  GeoJSON polygon file
    pole_points:   GeoJSON point file
    camera_table:  CSV with columns: camera_type, hfov_deg, range_m, unit_price_rm
    offset_table:  CSV with columns: offset

Outputs (dict of GeoDataFrames):
    dissolved_buildings, candidate_cctv, surv_area, aoi, hex_grid,
    hex_centroids, poles, cand_cctv_clean, wedge, camera_cost_summary
"""

import json
import csv
import math
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import (
    Point, Polygon, MultiPolygon, LineString, MultiLineString, mapping
)
from shapely.ops import unary_union
from shapely.validation import make_valid
from scipy.spatial import cKDTree


def run_cctv_pipeline(building_path, parking_path, poles_path, camera_csv_path, offset_csv_path):
    """
    Run the full CCTV planning pipeline.
    Returns dict of { layer_name: geojson_dict }
    """

    # ── Load inputs ──
    gdf_building = gpd.read_file(building_path)
    gdf_parking = gpd.read_file(parking_path)
    gdf_poles = gpd.read_file(poles_path)

    with open(camera_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        camera_rows = [r for r in reader]

    with open(offset_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        offset_rows = [r for r in reader]

    offsets = [float(r['offset']) for r in offset_rows]

    # =====================================================================
    # BRANCH A: Building candidates
    # Steps 1-6: Dissolve → Simplify → Fix → PolygonsToLines →
    #            ExtractVertices → DeleteDuplicateGeometries
    # =====================================================================

    # Step 1: Dissolve buildings
    dissolved_geom = unary_union(gdf_building.geometry)
    dissolved_geom = make_valid(dissolved_geom)
    gdf_dissolved = gpd.GeoDataFrame(geometry=[dissolved_geom], crs=gdf_building.crs)

    # Step 2: Simplify (tolerance=1 in QGIS model, METHOD=0 Douglas-Peucker)
    # FIX: The QGIS model uses tolerance=1 in the layer's CRS units.
    # For projected CRS (e.g. meters) this is 1m; for EPSG:4326 it's 1 degree.
    # Since the model uses degree-based parameters elsewhere (buffer=0.000269,
    # hspacing=2*0.0002695), the data is likely in EPSG:4326.
    # With tolerance=1 degree, simplify would essentially collapse the geometry,
    # which suggests the model was designed for a projected CRS or the tolerance
    # is intentionally aggressive. We use 1 here to match the model exactly.
    # If your data is in EPSG:4326, you may need to adjust this.
    simplified = dissolved_geom.simplify(1, preserve_topology=True)

    # Step 3: Fix geometries (METHOD=1 in QGIS = Structure method)
    fixed = make_valid(simplified)

    # Step 4: Polygons to lines
    def polygon_to_lines(geom):
        lines = []
        if geom.geom_type == 'Polygon':
            lines.append(LineString(geom.exterior.coords))
            for interior in geom.interiors:
                lines.append(LineString(interior.coords))
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                lines.extend(polygon_to_lines(poly))
        elif geom.geom_type == 'GeometryCollection':
            for g in geom.geoms:
                if g.geom_type in ('Polygon', 'MultiPolygon'):
                    lines.extend(polygon_to_lines(g))
        return lines

    building_lines = polygon_to_lines(fixed)

    # Step 5: Extract vertices
    vertices = set()
    for line in building_lines:
        for coord in line.coords:
            vertices.add((round(coord[0], 8), round(coord[1], 8)))

    # Step 6: Delete duplicate geometries (already deduped via set)
    candidate_points = [Point(v) for v in vertices]
    gdf_candidate_cctv = gpd.GeoDataFrame(
        geometry=candidate_points,
        crs=gdf_building.crs
    )

    # =====================================================================
    # BRANCH B: Merge buildings + parking → AOI → hex grid → centroids
    # Steps 7-14
    # =====================================================================

    # FIX: Match model3 order exactly:
    # Step 7: dissolved_buildings is already computed above (dissolve_1)
    # Step 8: Merge dissolved_buildings + parking_area (mergevectorlayers_1)
    merged_geoms = list(gdf_dissolved.geometry) + list(gdf_parking.geometry)
    gdf_merged = gpd.GeoDataFrame(geometry=merged_geoms, crs=gdf_building.crs)

    # Step 9: Dissolve the merged layer (dissolve_2) → surv_area
    surv_geom = unary_union(gdf_merged.geometry)
    surv_geom = make_valid(surv_geom)
    gdf_surv_area = gpd.GeoDataFrame(geometry=[surv_geom], crs=gdf_building.crs)

    # Step 10: Buffer → AOI (0.000269 degrees ≈ ~30m)
    # FIX: QGIS END_CAP_STYLE: 0=Round, 1=Flat, 2=Square
    # Shapely cap_style: 1=round, 2=flat, 3=square
    # QGIS JOIN_STYLE: 0=Round, 1=Miter, 2=Bevel
    # Shapely join_style: 1=round, 2=mitre, 3=bevel
    # Model3: END_CAP_STYLE=1 (Flat), JOIN_STYLE=1 (Miter), MITER_LIMIT=2
    aoi_geom = surv_geom.buffer(
        0.000269,
        cap_style=2,      # Flat (QGIS 1 → Shapely 2)
        join_style=2,      # Miter (QGIS 1 → Shapely 2)
        mitre_limit=2.0,
        resolution=5       # SEGMENTS=5
    )
    aoi_geom = make_valid(aoi_geom)
    gdf_aoi = gpd.GeoDataFrame(geometry=[aoi_geom], crs=gdf_building.crs)

    # Step 11: Create hexagonal grid matching QGIS native:creategrid TYPE=4
    hspacing = 2 * 0.0002695
    vspacing = 2 * 0.0002695
    hex_grid_polys = _create_hex_grid_qgis(aoi_geom.bounds, hspacing, vspacing)

    # Step 12: Clip grid by AOI
    gdf_hex_all = gpd.GeoDataFrame(geometry=hex_grid_polys, crs=gdf_building.crs)
    gdf_hex_grid = gpd.clip(gdf_hex_all, gdf_aoi)

    # Step 13: Centroids (ALL_PARTS=True)
    centroids = gdf_hex_grid.geometry.centroid
    gdf_hex_centroids = gpd.GeoDataFrame(geometry=centroids, crs=gdf_building.crs)

    # Step 14: Add geometry attributes (CALC_METHOD=1, ellipsoidal)
    # For EPSG:4326, xcoord=longitude, ycoord=latitude — same as .x/.y
    gdf_hex_centroids['xcoord'] = gdf_hex_centroids.geometry.x
    gdf_hex_centroids['ycoord'] = gdf_hex_centroids.geometry.y

    # Also add HubDist=0 field (fieldcalculator_3)
    gdf_hex_centroids['HubDist'] = 0.0

    # =====================================================================
    # BRANCH C: Pole filtering
    # Steps 16-17: poles within parking → poles within AOI
    # =====================================================================

    # Step 16: Extract by location - poles within parking (predicate=6 = "are within")
    gdf_poles_in_parking = gpd.sjoin(
        gdf_poles, gdf_parking, predicate='within', how='inner'
    ).drop(columns=['index_right'], errors='ignore')

    # Step 17: Extract by location - poles within AOI (predicate=6 = "are within")
    gdf_poles_filtered = gdf_poles_in_parking[
        gdf_poles_in_parking.geometry.within(aoi_geom)
    ].copy()
    gdf_poles_filtered = gdf_poles_filtered.reset_index(drop=True)

    # =====================================================================
    # Compute base_az for BOTH building candidates and pole candidates
    # =====================================================================

    centroid_coords = np.array([(p.x, p.y) for p in gdf_hex_centroids.geometry])

    if len(centroid_coords) > 0:
        tree = cKDTree(centroid_coords)
    else:
        tree = None

    def compute_base_az_and_expand(gdf_candidates, candidate_type):
        """For each candidate, find nearest hex centroid, compute base_az,
        then expand with offsets to create N rows per candidate.

        Matches QGIS flow:
          joinbynearest → fieldcalculator(base_az) → fieldcalculator(join_id=1)
          → joinattributestable(offsets) → fieldcalculator(azimuth)
        """
        if len(gdf_candidates) == 0 or tree is None:
            return gpd.GeoDataFrame(columns=['geometry', 'base_az', 'azimuth', 'type'])

        coords = np.array([(p.x, p.y) for p in gdf_candidates.geometry])
        _, idx = tree.query(coords)

        rows = []
        for i, (_, cand_row) in enumerate(gdf_candidates.iterrows()):
            nearest_hex = centroid_coords[idx[i]]
            base_az = _azimuth_degrees(
                cand_row.geometry.x, cand_row.geometry.y,
                nearest_hex[0], nearest_hex[1]
            )
            base_az = (base_az + 360) % 360

            for offset in offsets:
                az = (base_az + offset) % 360
                rows.append({
                    'geometry': cand_row.geometry,
                    'base_az': base_az,
                    'azimuth': az,
                    'offset': offset,
                    'type': candidate_type,
                })

        return gpd.GeoDataFrame(rows, crs=gdf_candidates.crs)

    # Building candidates with N azimuths (via joinbynearest_1 → fieldcalculator_15 → ...)
    gdf_building_3az = compute_base_az_and_expand(gdf_candidate_cctv, 'building')

    # Pole candidates with N azimuths (via joinbynearest_3 → fieldcalculator_10 → ...)
    gdf_pole_3az = compute_base_az_and_expand(gdf_poles_filtered, 'pole')

    # =====================================================================
    # Merge + assign camera type + join specs
    # =====================================================================

    # Step 30: Merge all candidates (mergevectorlayers_2)
    gdf_all_3az = pd.concat([gdf_pole_3az, gdf_building_3az], ignore_index=True)
    if len(gdf_all_3az) > 0:
        gdf_all_3az = gpd.GeoDataFrame(gdf_all_3az, crs=gdf_building.crs)
    else:
        gdf_all_3az = gpd.GeoDataFrame(
            columns=['geometry', 'base_az', 'azimuth', 'type', 'camera_type'],
            crs=gdf_building.crs
        )

    # FIX: Step 31 - Model3 HARDCODES camera_type = 'Type A' (fieldcalculator_4)
    if len(gdf_all_3az) > 0:
        gdf_all_3az['camera_type'] = 'Type A'

    # Step 35: Join camera specs by camera_type (joinattributestable_4)
    # Model3: first creates a cross-join of candidate_cctv × camera_table via join_id=1,
    # then joins merged_all to that by camera_type with METHOD=1 (first match only)
    cam_df = pd.DataFrame(camera_rows)
    for col in ['hfov_deg', 'range_m', 'unit_price_rm']:
        if col in cam_df.columns:
            cam_df[col] = pd.to_numeric(cam_df[col], errors='coerce')
    if 'camera_type' in cam_df.columns:
        cam_df['camera_type'] = cam_df['camera_type'].str.strip()

    if len(gdf_all_3az) > 0 and len(cam_df) > 0:
        gdf_all_specs = gdf_all_3az.merge(cam_df, on='camera_type', how='left')
        gdf_all_specs = gpd.GeoDataFrame(gdf_all_specs, crs=gdf_building.crs)
    else:
        gdf_all_specs = gdf_all_3az.copy()
        for col in ['hfov_deg', 'range_m', 'unit_price_rm']:
            if col not in gdf_all_specs.columns:
                gdf_all_specs[col] = 0

    # =====================================================================
    # Outputs: cand_cctv_clean, wedge, camera_cost_summary
    # =====================================================================

    # Refactor fields → cand_cctv_clean (refactorfields_1)
    clean_cols = ['geometry', 'azimuth', 'camera_type', 'hfov_deg', 'range_m', 'unit_price_rm']
    gdf_cand_clean = gdf_all_specs[[c for c in clean_cols if c in gdf_all_specs.columns]].copy()
    gdf_cand_clean['run_id'] = 'cctv_run'

    # Step 37: Wedge buffer (geometrybyexpression_2)
    # FIX: QGIS wedge_buffer does NOT apply cos(lat) correction.
    # It uses the outer_radius directly in CRS units:
    #   wedge_buffer($geometry, azimuth, hfov_deg, range_m / 111320)
    wedge_geoms = []
    wedge_attrs = []
    for _, row in gdf_all_specs.iterrows():
        az = float(row.get('azimuth', 0))
        hfov = float(row.get('hfov_deg', 90))
        range_m = float(row.get('range_m', 30))
        pt = row.geometry
        wedge = _wedge_buffer(pt.x, pt.y, az, hfov, range_m)
        wedge_geoms.append(wedge)
        wedge_attrs.append({
            'camera_type': row.get('camera_type', ''),
            'azimuth': az,
            'hfov_deg': hfov,
            'range_m': range_m,
            'unit_price_rm': row.get('unit_price_rm', 0),
        })

    gdf_wedge = gpd.GeoDataFrame(
        wedge_attrs,
        geometry=wedge_geoms,
        crs=gdf_building.crs
    ) if wedge_geoms else gpd.GeoDataFrame(columns=['geometry', 'camera_type'])

    # Step 36: Aggregate → Camera Cost Summary (native:aggregate_1)
    if len(gdf_all_specs) > 0 and 'camera_type' in gdf_all_specs.columns:
        cost_summary = gdf_all_specs.groupby('camera_type').agg(
            count=('azimuth', 'size'),
            unit_price_rm=('unit_price_rm', 'min'),
            total_cost_rm=('unit_price_rm', 'sum')
        ).reset_index()
    else:
        cost_summary = pd.DataFrame(columns=['camera_type', 'count', 'unit_price_rm', 'total_cost_rm'])

    # =====================================================================
    # Convert all to GeoJSON dicts
    # =====================================================================

    def to_geojson(gdf):
        if gdf is None or len(gdf) == 0:
            return {"type": "FeatureCollection", "features": []}
        gdf = gdf.copy()
        # Drop non-serializable columns
        for col in gdf.columns:
            if col != 'geometry' and gdf[col].dtype == 'object':
                gdf[col] = gdf[col].astype(str)
        return json.loads(gdf.to_json())

    def df_to_geojson(df):
        """Convert a plain DataFrame (no geometry) to a pseudo-GeoJSON for the frontend."""
        features = []
        for _, row in df.iterrows():
            features.append({
                "type": "Feature",
                "geometry": None,
                "properties": {k: _safe_val(v) for k, v in row.items()}
            })
        return {"type": "FeatureCollection", "features": features}

    results = {
        'dissolved_buildings': to_geojson(gdf_dissolved),
        'candidate_cctv': to_geojson(gdf_candidate_cctv),
        'surv_area': to_geojson(gdf_surv_area),
        'aoi': to_geojson(gdf_aoi),
        'hex_grid': to_geojson(gdf_hex_grid),
        'poles': to_geojson(gdf_poles_filtered),
        'cand_cctv_clean': to_geojson(gdf_cand_clean),
        'wedge': to_geojson(gdf_wedge),
        'camera_cost_summary': df_to_geojson(cost_summary),
    }

    return results


# =====================================================================
# Geometry helper functions
# =====================================================================

def _safe_val(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if pd.isna(v):
        return None
    return v


def _azimuth_degrees(x1, y1, x2, y2):
    """Compute azimuth in degrees from point1 to point2 (geographic coords).
    Matches QGIS azimuth() function: 0=North, clockwise."""
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dx, dy))
    return (angle + 360) % 360


def _wedge_buffer(lon, lat, azimuth_deg, hfov_deg, range_m):
    """Create a wedge/sector polygon.
    FIX: Matches QGIS wedge_buffer($geometry, azimuth, hfov_deg, range_m / 111320)
    QGIS does NOT apply cos(lat) correction — it treats the outer_radius
    uniformly in CRS units (degrees for EPSG:4326).
    """
    range_deg = range_m / 111320.0
    half_fov = hfov_deg / 2.0
    start_az = azimuth_deg - half_fov
    end_az = azimuth_deg + half_fov

    points = [(lon, lat)]  # apex
    steps = 32
    for i in range(steps + 1):
        az = start_az + (end_az - start_az) * i / steps
        az_rad = math.radians(az)
        # QGIS wedge_buffer: treats outer_radius uniformly in CRS units
        # azimuth convention: 0=North, clockwise → dx=sin(az), dy=cos(az)
        # NO cos(lat) correction to match QGIS behavior
        dx = range_deg * math.sin(az_rad)
        dy = range_deg * math.cos(az_rad)
        points.append((lon + dx, lat + dy))
    points.append((lon, lat))  # close

    return Polygon(points)


def _create_hex_grid_qgis(bounds, hspacing, vspacing):
    """Create hexagonal grid polygons matching QGIS native:creategrid TYPE=4.

    QGIS TYPE=4 creates flat-top hexagons where:
    - HSPACING = horizontal distance between hex centers in the same row
    - VSPACING = vertical distance between hex centers in adjacent rows
      (but QGIS internally computes the actual vertical offset)

    For TYPE=4 (Hexagon) in QGIS, the grid is built with:
    - Hex width = HSPACING
    - Hex height = VSPACING
    - Columns offset every other row by HSPACING/2
    - Rows spaced at VSPACING * 3/4
    """
    minx, miny, maxx, maxy = bounds

    # QGIS hex grid TYPE=4: flat-top hexagons
    # The hexagon dimensions are derived from spacing
    hex_width = hspacing   # full width of one hex
    hex_height = vspacing  # full height of one hex

    # Half dimensions for vertex computation
    half_w = hex_width / 2.0
    quarter_h = hex_height / 4.0

    # Vertical step between rows (3/4 of hex height for flat-top hex tiling)
    row_step = hex_height * 3.0 / 4.0

    polygons = []
    row = 0
    y = miny
    while y <= maxy + hex_height:
        # Offset every other row by half the hex width
        x_offset = half_w if (row % 2 == 1) else 0
        x = minx + x_offset
        while x <= maxx + hex_width:
            # Flat-top hexagon vertices (QGIS TYPE=4 convention)
            # Center at (x, y), vertices go clockwise from top
            hex_poly = Polygon([
                (x,            y + hex_height / 2.0),    # top center
                (x + half_w,   y + quarter_h),            # top right
                (x + half_w,   y - quarter_h),            # bottom right
                (x,            y - hex_height / 2.0),    # bottom center
                (x - half_w,   y - quarter_h),            # bottom left
                (x - half_w,   y + quarter_h),            # top left
                (x,            y + hex_height / 2.0),    # close
            ])
            polygons.append(hex_poly)
            x += hex_width
        y += row_step
        row += 1

    return polygons
