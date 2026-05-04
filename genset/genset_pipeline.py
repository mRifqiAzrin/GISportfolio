"""
genset_pipeline.py
==================
GIS pipeline that takes a cell site + a pre-fetched list of substations
(from the frontend via Overpass) and returns road distances + route polylines
for all substations reachable within 2 km via road network.

Overpass is handled in the frontend JS (same as the original index.html)
to avoid server-side header/auth conflicts with the Overpass API.

Usage:
    from genset_pipeline import route_substations
    results = route_substations(
        site_lat=3.1234, site_lng=101.5678,
        substations=[{"osm_id":"...", "name":"...", "lat":..., "lng":...}, ...]
    )
"""

import osmnx as ox
import networkx as nx
import logging
import time

logger = logging.getLogger(__name__)

# ── OSMnx config ─────────────────────────────────────────────────────────────
ox.settings.log_console = False
ox.settings.use_cache   = True   # caches road tiles — repeated calls in same area are fast

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_ROAD_DIST_M = 2000   # filter: only return substations within this road distance
GRAPH_BUFFER_M  = 2500   # road network download radius around the site


# ── Road network helpers ──────────────────────────────────────────────────────

def _get_road_graph(lat: float, lng: float, radius_m: int = GRAPH_BUFFER_M):
    """
    Download (or load from OSMnx cache) the undirected road network within
    radius_m metres of the given point.

    network_type='all' includes every road type — cable routing ignores
    traffic direction so we use the undirected graph.
    """
    G = ox.graph_from_point(
        (lat, lng),
        dist=radius_m,
        network_type="all",
        simplify=True,
    )
    return ox.convert.to_undirected(G)


def _nearest_node(G, lat: float, lng: float):
    return ox.distance.nearest_nodes(G, X=lng, Y=lat)


def _route_to_coords(G, route_nodes: list) -> list:
    """Convert a node-ID path to [[lat, lng], ...] for Leaflet polylines."""
    return [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in route_nodes]


# ── Main public function ──────────────────────────────────────────────────────

def route_substations(
    site_lat: float,
    site_lng: float,
    substations: list,
    max_road_dist_m: int = MAX_ROAD_DIST_M,
    graph_buffer_m:  int = GRAPH_BUFFER_M,
) -> dict:
    """
    For each substation in the list, compute the shortest road distance
    from the site and return those within max_road_dist_m metres.

    Parameters
    ----------
    site_lat, site_lng  : WGS-84 coordinates of the cell site
    substations         : list of dicts with keys: osm_id, name, lat, lng
    max_road_dist_m     : road distance threshold in metres (default 2000)
    graph_buffer_m      : OSMnx download radius (default 2500)

    Returns
    -------
    {
        "site":                   {"lat": float, "lng": float},
        "results":                [...],
        "substations_checked":    int,
        "substations_within_2km": int,
        "error":                  str | None,
        "elapsed_s":              float
    }

    Each result dict:
    {
        "name":         str,
        "lat":          float,
        "lng":          float,
        "osm_id":       str,
        "road_dist_m":  float,
        "road_dist_km": float,
        "route_coords": [[lat, lng], ...]
    }
    """
    t0 = time.time()

    if not substations:
        return {
            "site":                   {"lat": site_lat, "lng": site_lng},
            "results":                [],
            "substations_checked":    0,
            "substations_within_2km": 0,
            "error":                  "No substations provided",
            "elapsed_s":              0,
        }

    # 1. Download road network
    try:
        G = _get_road_graph(site_lat, site_lng, graph_buffer_m)
    except Exception as e:
        return {
            "site":                   {"lat": site_lat, "lng": site_lng},
            "results":                [],
            "substations_checked":    len(substations),
            "substations_within_2km": 0,
            "error":                  f"Road network download failed: {e}",
            "elapsed_s":              round(time.time() - t0, 2),
        }

    # 2. Snap site to nearest graph node
    try:
        site_node = _nearest_node(G, site_lat, site_lng)
    except Exception as e:
        return {
            "site":                   {"lat": site_lat, "lng": site_lng},
            "results":                [],
            "substations_checked":    len(substations),
            "substations_within_2km": 0,
            "error":                  f"Could not snap site to road network: {e}",
            "elapsed_s":              round(time.time() - t0, 2),
        }

    # 3. Route to each substation
    results = []
    for sub in substations:
        try:
            sub_node    = _nearest_node(G, sub["lat"], sub["lng"])
            road_dist_m = nx.shortest_path_length(G, site_node, sub_node, weight="length")

            if road_dist_m > max_road_dist_m:
                continue

            route_nodes  = nx.shortest_path(G, site_node, sub_node, weight="length")
            route_coords = _route_to_coords(G, route_nodes)

            results.append({
                "name":         sub.get("name", "Substation"),
                "lat":          sub["lat"],
                "lng":          sub["lng"],
                "osm_id":       sub.get("osm_id", ""),
                "road_dist_m":  round(road_dist_m, 1),
                "road_dist_km": round(road_dist_m / 1000, 3),
                "route_coords": route_coords,
            })

        except nx.NetworkXNoPath:
            logger.debug(f"No path to {sub.get('name')}")
        except nx.NodeNotFound:
            logger.debug(f"Node not found for {sub.get('name')}")
        except Exception as e:
            logger.warning(f"Routing error for {sub.get('name')}: {e}")

    results.sort(key=lambda x: x["road_dist_m"])
    elapsed = round(time.time() - t0, 2)

    logger.info(
        f"Pipeline done in {elapsed}s — "
        f"{len(results)}/{len(substations)} substations within {max_road_dist_m}m"
    )

    return {
        "site":                   {"lat": site_lat, "lng": site_lng},
        "results":                results,
        "substations_checked":    len(substations),
        "substations_within_2km": len(results),
        "error":                  None,
        "elapsed_s":              elapsed,
    }


# ── CLI quick-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, sys
    _lat = float(sys.argv[1]) if len(sys.argv) > 1 else 3.1478
    _lng = float(sys.argv[2]) if len(sys.argv) > 2 else 101.6953
    # Dummy substation for smoke-test
    _subs = [{"osm_id": "test", "name": "Test Sub", "lat": _lat + 0.005, "lng": _lng + 0.005}]
    print(json.dumps(route_substations(_lat, _lng, _subs), indent=2))
