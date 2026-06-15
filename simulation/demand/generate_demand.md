# Demand Data Sources — Michigan Ave & Livernois

## Intersection

**Michigan Avenue & Livernois Avenue, Detroit, MI**  
Coordinates: approximately 42.3326° N, 83.1274° W

## AADT Estimates

| Road | AADT (vpd) | Source | Year |
| --- | --- | --- | --- |
| Michigan Avenue | ~20,000 | MDOT Traffic Volumes Open Data | 2022 |
| Livernois Avenue | ~10,000 | MDOT Traffic Volumes Open Data | 2022 |

## Sources

- **MDOT Traffic Volumes Open Data Portal**
  [https://gis-michigan.opendata.arcgis.com/datasets/mdot::2022-traffic-volumes]
  Provides AADT and CAADT by road segment across Michigan.

- **SEMCOG Regional Traffic Counts Database**  
  [https://data.semcog.org/traffic-counts]  
  24-hour counts for Southeast Michigan, adjusted with seasonal factors.

- **Detroit Open Data Portal — Traffic Volumes**  
  [https://data.detroitmi.gov/maps/049ffe8e321b4a70a2b09dd66b9e0255]  
  City of Detroit AADT/CAADT split by year, derived from MDOT.

## Methodology

Peak hour volumes were derived from AADT using a standard urban arterial
K-factor of 0.09 (peak hour = 9% of AADT). This is consistent with MDOT
and ITE (Institute of Transportation Engineers) guidelines for urban
corridors in Southeast Michigan.

Off-peak volumes estimated at 44% of peak hour volume, consistent with
typical urban arterial diurnal profiles.

Directional splits:

- Michigan Ave: 55/45 WB/EB in AM peak (commuters heading downtown),
  reversed (45/55) in PM peak.
- Livernois: 50/50 both directions (balanced residential/commercial corridor).

## To Update with Real Count Data

1. Visit [https://data.semcog.org/traffic-counts]
2. Search for "Michigan Ave" near "Livernois" in Detroit
3. Download the 24-hour count report (CSV or PDF)
4. Update the `AADT` dict in `generate_demand.py` with the actual figures
5. Run `python simulation/demand/generate_demand.py` to regenerate `routes.rou.xml`
