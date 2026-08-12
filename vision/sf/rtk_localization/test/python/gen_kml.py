import csv
import simplekml

kml = simplekml.Kml()
coords = []

with open("gnss_global_pose_latlon.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        coords.append((lon, lat))  # KML = (lon, lat)

linestring = kml.newlinestring(
    name="GPS Path",
    coords=coords
)

# Optional styling
linestring.style.linestyle.width = 4
linestring.style.linestyle.color = simplekml.Color.red

kml.save("gnss_global_pose_latlon_k.kml")
