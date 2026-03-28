import folium

def create_map(points):
    if not points:
        return None

    m = folium.Map(location=points[0], zoom_start=14)

    for lat, lon in points:
        folium.Marker(
            [lat, lon],
            popup="Pothole",
            icon=folium.Icon(color="red")
        ).add_to(m)

    return m