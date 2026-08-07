from typing import Any, Optional
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterEnum,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingMultiStepFeedback,
)
import psycopg2


class Atom5(QgsProcessingAlgorithm):
    def initAlgorithm(self, config: Optional[dict[str, Any]] = None):
        conn = psycopg2.connect(
            dbname='Qgis',
            user='Qgis',
            password='123456',
            host='localhost',
            port='5432'
        )
        cursor = conn.cursor()

        def fetch_tables(prefix):
            cursor.execute(f"""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public' AND tablename LIKE '{prefix}%';
            """)
            return [row[0] for row in cursor.fetchall()]

        self.mr_tables = fetch_tables('mr')
        self.polygon_tables = fetch_tables('polygon')
        self.population_tables = fetch_tables('population')

        cursor.close()
        conn.close()

        self.addParameter(QgsProcessingParameterEnum('mr', 'MR Table', self.mr_tables))
        self.addParameter(QgsProcessingParameterEnum('polygon', 'Polygon Table', self.polygon_tables))
        self.addParameter(QgsProcessingParameterEnum('population', 'Population Table', self.population_tables))

    def processAlgorithm(self, parameters: dict[str, Any], context: QgsProcessingContext, model_feedback: QgsProcessingFeedback) -> dict[str, Any]:
        feedback = QgsProcessingMultiStepFeedback(22, model_feedback)

        mr_table = self.mr_tables[parameters['mr']]
        polygon_table = self.polygon_tables[parameters['polygon']]
        population_table = self.population_tables[parameters['population']]

        feedback.pushInfo("[INFO] Connecting to PostGIS...")
        conn = psycopg2.connect(
            dbname='Qgis',
            user='Qgis',
            password='123456',
            host='localhost',
            port='5432'
        )
        cursor = conn.cursor()

        sql_steps = [
            # Step 0 - Load population
            f"""DROP TABLE IF EXISTS atom_population;
                CREATE TEMPORARY TABLE atom_population AS SELECT * FROM {population_table};""",

            # Step 1 - Load polygon
            f"""DROP TABLE IF EXISTS atom_polygon;
                CREATE TABLE atom_polygon AS SELECT * FROM {polygon_table};""",

            # Step 2 - Load MR
            f"""DROP TABLE IF EXISTS atom_mr;
                CREATE TEMPORARY TABLE atom_mr AS SELECT * FROM {mr_table};""",

            # Step 3 - Refactor population
            f"""DROP TABLE IF EXISTS atom_refactored_population;
                CREATE TEMPORARY TABLE atom_refactored_population AS
                SELECT population, area_before, ST_Transform(geom, 4326) AS geom FROM atom_population;""",

            # Step 4 - Buffer
            f"""DROP TABLE IF EXISTS atom_mrgrid;
                CREATE TEMPORARY TABLE atom_mrgrid AS
                SELECT (ST_Dump(ST_Buffer(geom, 0.001, 'endcap=square join=mitre mitre_limit=1 quad_segs=1'))).geom::geometry(Polygon, 4326) AS geom
                FROM atom_mr;""",

            # Step 5 - Refactor Region
            f"""DROP TABLE IF EXISTS atom_refactored_state_fields;
                CREATE TEMPORARY TABLE atom_refactored_state_fields AS
                SELECT
                    "State",
                    CASE WHEN "Region" = 'NORTHERN' AND "State" = 'PAHANG' THEN 'EASTERN' ELSE "Region" END AS "Region",
                    geom
                FROM atom_polygon;""",

            # Step 6 - DBSCAN
            f"""DROP TABLE IF EXISTS atom_dbscan_clusters;
                CREATE TABLE atom_dbscan_clusters AS
                SELECT *, COUNT(*) OVER (PARTITION BY cluster_id) AS cluster_size
                FROM (
                    SELECT *, ST_ClusterDBSCAN(geom, eps := 0.002, minpoints := 15) OVER () AS cluster_id
                    FROM atom_mr
                ) AS clustered;""",

            # Step 7 - Export Buffer to table
            f"""DROP TABLE IF EXISTS mr_grid;
                CREATE TABLE mr_grid AS SELECT * FROM atom_mrgrid;""",

            # Step 8 - Filter DBSCAN clusters
            f"""DROP TABLE IF EXISTS dbscan_cleaned;
                CREATE TABLE dbscan_cleaned AS
                SELECT * FROM atom_dbscan_clusters WHERE cluster_id IS NOT NULL;""",

            # Step 9 - Bounding Geometry (Convex Hulls)
            f"""DROP TABLE IF EXISTS atom_bounding_geometries;
                CREATE TABLE atom_bounding_geometries AS
                SELECT cluster_id,
                        CASE
                           WHEN GeometryType(ST_ConvexHull(ST_Collect(geom))) IN ('POLYGON', 'MULTIPOLYGON') THEN ST_ConvexHull(ST_Collect(geom))::geometry(Polygon, 4326)
                           ELSE NULL  -- or ST_Buffer(..., 0) as a fallback
                        END AS geom
                FROM dbscan_cleaned
                GROUP BY cluster_id;""",

            # Step 10 - Refactor hulls
            f"""DROP TABLE IF EXISTS convex_fields;
                CREATE TEMPORARY TABLE convex_fields AS
                SELECT row_number() OVER () AS id, cluster_id, geom FROM atom_bounding_geometries;""",

            # Step 11 - Intersection with Region
            f"""DROP TABLE IF EXISTS region_intersection;
                CREATE TEMPORARY TABLE region_intersection AS
                SELECT 
                    a.cluster_id, a.id,
                    b."Region", b."State",
                ST_Intersection(a.geom, b.geom) AS geom
                FROM convex_fields a
                JOIN atom_refactored_state_fields b ON ST_Intersects(a.geom, b.geom);""",

            # Step 12 - Central
            f"""DROP TABLE IF EXISTS central_extract;
                CREATE TEMPORARY TABLE central_extract AS
                SELECT * FROM region_intersection WHERE "Region" = 'CENTRAL';""",

            # Step 13 - Refactor central
            f"""DROP TABLE IF EXISTS central_refactored;
                CREATE TEMPORARY TABLE central_refactored AS
                SELECT cluster_id, "State", "Region", geom FROM central_extract;""",

            # Step 14 - Northern
            f"""DROP TABLE IF EXISTS northern_extract;
                CREATE TEMPORARY TABLE northern_extract AS
                SELECT * FROM region_intersection WHERE "Region" = 'NORTHERN';""",

            # Step 15 - Intersect Population
            f"""DROP TABLE IF EXISTS intersect_pop_convex;
                CREATE TEMPORARY TABLE intersect_pop_convex AS
                SELECT
                    a.population,
                    a.area_before,
                    b.cluster_id,
                    b."Region",
                    b."State",
                    ST_Intersection(a.geom, b.geom) AS geom
                FROM atom_refactored_population a
                JOIN central_refactored b ON ST_Intersects(a.geom, b.geom);""",


            # Step 16 - Eastern
            f"""DROP TABLE IF EXISTS eastern_extract;
                CREATE TEMPORARY TABLE eastern_extract AS
                SELECT * FROM region_intersection WHERE "Region" = 'EASTERN';""",

            # Step 17 - Calculate area_after
            f"""DROP TABLE IF EXISTS intersect_area;
                CREATE TEMPORARY TABLE intersect_area AS
                SELECT *,
                    ST_Area(ST_Transform(geom, 3857)) AS area_after
                FROM intersect_pop_convex;""",

            # Step 18 - Southern
            f"""DROP TABLE IF EXISTS southern_extract;
                CREATE TEMPORARY TABLE southern_extract AS
                SELECT * FROM region_intersection WHERE "Region" = 'SOUTHERN';""",

            # Step 19 - Calculate % area
            f"""DROP TABLE IF EXISTS percent_;
                CREATE TEMPORARY TABLE percent_area AS
                SELECT *, (area_after / area_before) * 100.0 AS percentage
                FROM intersect_area;""",

            # Step 20 - Calculate Real Population
            f"""DROP TABLE IF EXISTS final_population;
                CREATE TEMPORARY TABLE final_population AS
                SELECT *, ROUND(population * (percentage / 100.0))::integer AS Real_population
                FROM percent_area;""",

            # Step 21 - Final export
            f"""DROP TABLE IF EXISTS population_cluster;
                CREATE TABLE population_cluster AS SELECT * FROM final_population;"""
        ]

        for step, sql in enumerate(sql_steps):
            feedback.setCurrentStep(step)
            if feedback.isCanceled():
                conn.rollback()
                cursor.close()
                conn.close()
                return {}
            feedback.pushInfo(f"[INFO] Step {step + 1}/22 executing...")
            cursor.execute(sql)

        conn.commit()
        cursor.close()
        conn.close()
        feedback.pushInfo("[INFO] ATOM5 process completed successfully.")

        return {}

    def name(self) -> str:
        return 'ATOM5'

    def displayName(self) -> str:
        return 'ATOM5'

    def group(self) -> str:
        return ''

    def groupId(self) -> str:
        return ''

    def createInstance(self):
        return Atom5()
