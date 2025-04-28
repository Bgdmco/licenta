import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from functions import nx_to_ig, greedy_triangulation_routing, mst_routing, fill_holes, extract_relevant_polygon, osm_to_ig
import igraph as ig
import networkx as nx
import shapely
import numpy as np 
from matplotlib import cm
import pandas as pd
place_name = "Bucharest, Romania"


G_bike = ox.graph_from_place(place_name, network_type="bike", retain_all=True)
edges = ox.graph_to_gdfs(G_bike, nodes=False)
bike_lanes = edges[edges["highway"].isin(["cycleway"])]



location = ox.geocoder.geocode_to_gdf(place_name)
location = fill_holes(extract_relevant_polygon(place_name, shapely.geometry.shape(location['geometry'][0])))
debug = False
if debug is True:
    try:
        color = cm.rainbow(np.linspace(0,1,len(location)))
        for poly,c in zip(location, color):
            plt.plot(*poly.exterior.xy, c = c)
            for intr in poly.interiors:
                plt.plot(*intr.xy, c = "red")
    except:
        plt.plot(*location.exterior.xy)
    plt.show()

G = ox.graph_from_polygon(location, network_type="drive", retain_all= True, simplify= True)
#fig, ax = ox.plot_graph(G, show=True, node_size=10, edge_linewidth=0.5)
largest_component = max(nx.weakly_connected_components(G), key=len)
G_largest = G.subgraph(largest_component).copy()
fig, ax = ox.plot_graph(G_largest, show=True, node_size=10, edge_linewidth=0.5)


nodes = list(G_largest.nodes(data=True))
nodes_to_use = {}
osmid_list = []
x_list = []
y_list = []
for node, attr in (nodes):
    osmid_list.append(node)
    x_list.append(attr['x'])
    y_list.append(attr['y'])
nodes_to_use.update({"osmid" : osmid_list, "x": x_list, "y": y_list})
nodes_df = pd.DataFrame(nodes_to_use)

edges = list(G_largest.edges(data = True))
u_list = []
v_list = []
length_list = []
edge_osmid_list = []
edges_to_use = {}
for node1, node2, attr in edges:
    u_list.append(node1)
    v_list.append(node2)
    length_list.append(attr['length'])
    if isinstance(attr['osmid'], list):
        edge_osmid_list.append(max(attr['osmid']))
    else:
        edge_osmid_list.append(attr['osmid'])
edges_to_use.update({"osmid":edge_osmid_list, 'u': u_list, "v": v_list, 'length': length_list})
edges_df = pd.DataFrame(edges_to_use)

G_carall = osm_to_ig(nodes_df, edges_df)




top_stations = ["Gara de Nord 1", "Basarab 1 - M1", "Obor", "Piața Sudului", "Titan", "Piața Unirii 1", "Dristor 1", "Eroilor", "Crângași", "Piața Victoriei 1", "Eroii Revoluției", "Politehnica", "Lujerului"]

metro_stations_gdf = ox.features_from_place(
    place_name, tags={"railway": "station", "subway": "yes"}
)
metro_stations_gdf = metro_stations_gdf[metro_stations_gdf["subway"] == "yes"]
metro_stations_gdf = metro_stations_gdf[metro_stations_gdf.geometry.apply(lambda geom: isinstance(geom, Point))]
metro_stations_gdf = metro_stations_gdf[metro_stations_gdf["public_transport"] == "station"]
metro_stations_gdf = metro_stations_gdf[metro_stations_gdf["name"].isin(top_stations)]
#print(metro_stations_gdf)


top_routes = ["Drumul Taberei", "Bulevardul Iuliu Maniu", "Splaiul Independenței", "Bulevardul Voluntarilor", "Bulevardul Unirii", "Splaiul Unirii", "Piața Victoriei", "Piața Unirii", "Piața Romana", "Bulevardul Eroii Sanitari", "Bulevardul Eroilor", "Calea Moșilor", "Șoseaua Mihai Bravu", "Șoseaua Ștefan cel Mare"]
top_streets = ["Bulevardul Catargiu", "Bulevardul Magheru", "Bulevardul Bălcescu", "Bulevardul Brătianu", "Șoseaua Ștefan cel Mare", "Splaiul Unirii", "Bulevardul Unirii", "Șoseaua Mihai Bravu", "Bulevardul Iuliu Maniu",  "Splaiul Independenței", "Bulevardul Regina Elisabeta", "Bulevardul Mihail Kogălniceanu", "Drumul Taberei", "Calea Moșilor", "Șoseaua Colentina", "Strada Barbu Văcărescu", "Bulevardul Carol I"]
street_network = ox.graph_from_place(place_name, network_type="drive", retain_all=True)
streets_gdf  = ox.graph_to_gdfs(street_network, nodes=False)
streets_gdf = streets_gdf[streets_gdf["name"].isin(top_streets)]
#print(streets_gdf)

points_of_interest = [
    {"name": "Pipera", "lat": 44.50585, "lon": 26.13702},
    {"name": "Baneasa", "lat": 44.49456, "lon": 26.07914},
    {"name": "Piata Victoriei", "lat": 44.45247, "lon": 26.08583},
    {"name": "Politehnica", "lat": 44.44437, "lon": 26.05265},
    {"name": "Jiului", "lat": 44.48249, "lon": 26.04104},
    {"name": "Palatul Parlamentului", "lat": 44.42754, "lon": 26.08785},
    {"name": "Chiajna", "lat": 44.45782, "lon": 25.97450},
    {"name": "Institutul de fizica", "lat": 44.348957, "lon": 26.03074},
    {"name": "NordEst Logistic Park", "lat": 44.48592, "lon": 26.21884},
    {"name": "Pantelimon", "lat": 44.44811, "lon": 26.21096},
    {"name": "Icme Ecab", "lat": 44.42034, "lon": 26.21877},
    {"name": "Danubiana", "lat": 44.36317, "lon": 26.19406},
    {"name": "Universitatea din Bucuresti", "lat": 44.43553, "lon": 26.10222},
    {"name": "Universitatea Politehnica", "lat": 44.43855, "lon": 26.04958},
    {"name": "ASE", "lat": 44.44475, "lon": 26.09778},
    {"name": "AFI PALACE", "lat": 44.43099, "lon": 26.05433},
    {"name": "Baneasa Shopping City", "lat": 44.50794, "lon": 26.09133},
    {"name": "Plaza Romania", "lat": 44.42854, "lon": 26.03352},
]

point_of_interest_gdf = gpd.GeoDataFrame(
    points_of_interest, 
    geometry=[Point(d["lon"], d["lat"]) for d in points_of_interest], 
    crs="EPSG:4326"
)




metro_stations_nodes = [ox.distance.nearest_nodes(G_largest, row.geometry.x, row.geometry.y) for _, row in metro_stations_gdf.iterrows()]
poi_nodes = [ox.distance.nearest_nodes(G_largest, d["lon"], d["lat"]) for d in points_of_interest]
nnids = metro_stations_nodes + poi_nodes
#node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
#for i, node in enumerate(nnids):
#    nnids[i] = node_mapping[node]

print(nnids)
GTs, GT_abstracts = greedy_triangulation_routing(G_carall, nnids)
#MST, MST_abstract = mst_routing(G_carall, nnids)
print(f"aici e printu {GTs}")
ig.plot(GTs, target=plt.gca())
plt.show()
G_combined = G.copy()

#def add_pois_to_graph(G, pois, node_type):
#    for idx, row in pois.iterrows():
#        node_id = f"{node_type}_{idx}"  # Unique node ID
#        G.add_node(node_id, x=row.geometry.x, y=row.geometry.y, name=row["name"], type=node_type)
#    return G
#
#G_combined = add_pois_to_graph(G_combined, metro_stations_gdf, "metro_stations")
#G_combined = add_pois_to_graph(G_combined, point_of_interest_gdf, "POI")
#
#print(f"Final Graph: {len(G_combined.nodes)} nodes, {len(G_combined.edges)} edges")
#fig, ax = plt.subplots(figsize=(10, 10))
#ox.plot_graph(G_combined, ax=ax, node_size=0, edge_color="lightgray", edge_linewidth=0.5, bgcolor="white", show=False)
#bike_lanes.plot(ax=ax, color="blue", linewidth=1.5, label="Bike Lanes")
#metro_stations_gdf.plot(ax=ax, color="red", markersize=10, label="Metro Stations")
##streets_gdf.plot(ax = ax, color = "green", linewidth=1.5, label = "Top Streets")
#point_of_interest_gdf.plot(ax=ax, color="purple", markersize=10, label="Employment Hubs/Univerity Centers/Shopping Centers")


#plt.legend()
#plt.title("Bike Lanes in Bucharest")
#plt.show()

#TODO

##Population density pe cartiere ca metric pentru a vedea cati oameni au acces la pista de biciclete-worldpop.org - DONE-TREBUIE DOAR PLOTUIT PE GRAPH
##Lungimea graphului cu pistele existente + rutele pe care le vor biciclestii-DONE
##GT graph pana la lungimea graphului de mai sus-DONE
##Metrics de la michael cu distanta parcursa cu masina din punctu A-B vs cu bicicleta-DONE
##Acoperirea pistei de biciclete ca arie fata de oras-DONE
##EDA pentru street network si pentru pista+dorinta biciclisti(nr noduri + edges, lungime, toate nodurile nu doar capetele edge-urilor)