# Module 2 Python 5 of 5 - Capstone project.
# Hands-on Lab: Interactive Visual Analytics with Folium
# Guillermo Dominguez

import folium
# https://www.python-graph-gallery.com/312-add-markers-on-folium-map

import wget
import pandas as pd

# Import folium MarkerCluster plugin
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon

# Show all columns in the PyCharm preview
pd.set_option('display.width', 800)  # avoid truncated view
pd.set_option('display.max_columns', 50)  # columns shown
pd.set_option('display.max_rows', 999)  # rows shown

# =====
# Task 1: Mark all launch sites on a map

# Download and read the `spacex_launch_geo.csv: an augmented dataset with latitude and longitude added for each site.
spacex_csv_file = wget.download('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv')
spacex_df = pd.read_csv(spacex_csv_file)
# print(spacex_df.head())

# Select relevant sub-columns: `Launch Site`, `Lat(Latitude)`, `Long(Longitude)`, `class`
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
print(launch_sites_df)

# Root path for the map folder
root_path = r'C:\Users\guillermo.dominguez\PycharmProjects\edx Python course'

# Start location is NASA Johnson Space Center
nasa_coordinate = [29.559684888503615, -95.0830971930759]

# Create map instance
site_map = folium.Map(location=nasa_coordinate, zoom_start=5, control_scale=True)

# Map title
# title_name = 'Map of NASA JSC and Launch sites'
title_name = 'Map of NASA JSC, Launch sites and Launch outcomes'
title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(title_name)
site_map.get_root().html.add_child(folium.Element(title_html))

# Create a blue circle at NASA Johnson Space Center's coordinate with a popup label showing its name
circle = folium.Circle(nasa_coordinate, radius=1000, color='#0016d3', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
# Create a blue circle at NASA Johnson Space Center's coordinate with an icon showing its name
marker = folium.map.Marker(
    nasa_coordinate,
    # Create an icon as a text label
    icon=DivIcon(icon_size=(50, 50), icon_anchor=(0,0), html='<div style="font-size: 12; color:#0016d3;"><b>%s</b></div>' % 'NASA JSC',))

site_map.add_child(circle)
site_map.add_child(marker)

# Now add  the Launch sites looking up on the small table
for i in range(0, len(launch_sites_df)):
    coordinate = [launch_sites_df.iloc[i]['Lat'], launch_sites_df.iloc[i]['Long']]
    name_circle = launch_sites_df['Launch Site'].iloc[i]
    circle = folium.Circle(coordinate, radius=100, color='#0016d3', fill=True).add_child(folium.Popup(name_circle))
    mark = folium.map.Marker(coordinate, icon=DivIcon(icon_size=(20, 20), icon_anchor=(0, 0),
                                                html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % name_circle, ))
    site_map.add_child(circle)
    site_map.add_child(mark)
site_map.save(root_path+'\\site_map.html', 'r')

# Task 2: Mark the success/failed launches for each site on the map
# Create new map instance
# site_map_cluster = folium.Map(location=nasa_coordinate, zoom_start=5, control_scale=True)

# Map title
# title_name_cluster = 'Map of launch outcomes clustered by site'
# title_html = '''
#              <h3 align="center" style="font-size:16px"><b>{}</b></h3>
#              '''.format(title_name_cluster)
# site_map_cluster.get_root().html.add_child(folium.Element(title_html))
site_map.get_root().html.add_child(folium.Element(title_html))

# Create a new column in launch_sites dataframe called marker_color to store the marker colors based on the class value
spacex_df.loc[spacex_df['class'] == 1, 'marker_color'] = 'green'
spacex_df.loc[spacex_df['class'] == 0, 'marker_color'] = 'red'
# print(spacex_df[['class', 'marked_color']])
print(spacex_df.tail(10))

# Marker clusters can be a good way to simplify a map containing many markers having the same coordinate
# Create a folium marker cluster
# marker_cluster = MarkerCluster().add_to(site_map_cluster)
marker_cluster = MarkerCluster().add_to(site_map)

# For each launch result in spacex_df data frame, add a folium.Marker to marker_cluster
# Add marker_cluster to current site_map
# site_map_cluster.add_child(marker_cluster)
site_map.add_child(marker_cluster)

# For each row in spacex_df data frame create a Marker object with its coordinate and customize the Marker's
# icon property to indicate if this launch was succeeded or failed,
for index, row in spacex_df.iterrows():
    icon_descrp = folium.Icon(color=row['marker_color'], icon_color=row['marker_color'], popup=row['class'])
    coordinate = [row['Lat'], row['Long']]
    folium.Marker(coordinate, icon=icon_descrp).add_to(marker_cluster)

# site_map_cluster.save(root_path+'\\site_map_cluster.html', 'r')

# TASK 3: Calculate the distances between a launch site to its proximities
# add a MousePosition on the map to get coordinate for a mouse over a point on the map.
# As such, while you are exploring the map, you can easily find the coordinates of any points of interests (such as railway)

# Add Mouse Position to get the coordinate (Lat, Long) for a mouse over on the map
formatter = "function(num) {return L.Util.formatNum(num, 5) + ' º ';};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,)

site_map.add_child(mouse_position)

# Function to calculate distances based on their coordinates
from math import sin, cos, sqrt, atan2, radians

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# find coordinates of the closet coastline
coastline_lat = 28.56297
coastline_lon = -80.56785
coordinate_coast = [coastline_lat, coastline_lon]
launch_site_lat = launch_sites_df.iloc[1]['Lat']
launch_site_long = launch_sites_df.iloc[1]['Long']

distance_coastline = calculate_distance(launch_site_lat, launch_site_long, coastline_lat, coastline_lon)
print("Distance of Launch site", launch_sites_df.iloc[1]['Launch Site'], "to coast is:", round(distance_coastline, 3), 'km')
# print(distance_coastline)

# After obtained its coordinate, create a folium.Marker to show the distance
# Display the distance between coastline point and launch site using the icon property
distance_marker = folium.Marker(
   coordinate_coast,
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_coastline),
       )
   )
site_map.add_child(distance_marker)

# Draw a PolyLine between a launch site to the selected coastline point
line_coast = folium.PolyLine(locations=[[launch_site_lat, launch_site_long], coordinate_coast], weight=2)
site_map.add_child(line_coast)

# Railway, Highway, City coordinates
Railway_lat = 28.57212
Railway_lon = -80.58527
coordinate_Railway = [Railway_lat, Railway_lon]
line_railway = folium.PolyLine(locations=[[launch_site_lat, launch_site_long], coordinate_Railway], weight=2)
site_map.add_child(line_railway)

distance_railway = calculate_distance(launch_site_lat, launch_site_long, Railway_lat, Railway_lon)
print("Distance of Launch site", launch_sites_df.iloc[1]['Launch Site'], "to closest railway is:", round(distance_railway, 3), 'km')

Highway_lat = 28.56334
Highway_lon = -80.57079
coordinate_Highway = [Highway_lat, Highway_lon]
line_highway = folium.PolyLine(locations=[[launch_site_lat, launch_site_long], coordinate_Highway], weight=2)
site_map.add_child(line_highway)

distance_highway = calculate_distance(launch_site_lat, launch_site_long, Highway_lat, Highway_lon)
print("Distance of Launch site", launch_sites_df.iloc[1]['Launch Site'], "to closest highway is:", round(distance_highway, 3), 'km')

city_lat = 28.38716
city_lon = -80.60201
coordinate_city = [city_lat, city_lon]
line_city = folium.PolyLine(locations=[[launch_site_lat, launch_site_long], coordinate_city], weight=2)
site_map.add_child(line_city)

distance_city = calculate_distance(launch_site_lat, launch_site_long, city_lat, city_lon)
print("Distance of Launch site", launch_sites_df.iloc[1]['Launch Site'], "to closest city (Cabo Canaveral) is:", round(distance_city, 3), 'km')

# Finally, save map as .html
site_map.save(root_path+'\\site_map_Launches.html', 'r')

print("End of Hand-on lab Interactive Visual Analytics with Folium")