import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from functions import fill_holes, extract_relevant_polygon, ox_to_csv
import igraph as ig
import networkx as nx
import shapely
import numpy as np 
from matplotlib import cm
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path


data_pth = Path(".\Data")
network_data_pth = data_pth/"network_data"
osmnxparameters = {'car30': {'network_type':'drive', 'custom_filter':'["maxspeed"~"^30$|^20$|^15$|^10$|^5$|^20 mph|^15 mph|^10 mph|^5 mph"]', 'export': True, 'retain_all': True},
                   'carall': {'network_type':'drive', 'custom_filter': None, 'export': True, 'retain_all': False},
                   'toproutes' :{'network_type':'bike', 'custom_filter':None, 'export': False, 'retain_all': True},
                   'bike_cyclewaytrack': {'network_type':'bike', 'custom_filter':'["cycleway"~"track"]', 'export': False, 'retain_all': True},
                   'bike_highwaycycleway': {'network_type':'bike', 'custom_filter':'["highway"~"cycleway"]', 'export': False, 'retain_all': True},
                   'bike_designatedpath': {'network_type':'all', 'custom_filter':'["highway"~"path"]["bicycle"~"designated"]', 'export': False, 'retain_all': True},
                   'bike_cyclewayrighttrack': {'network_type':'bike', 'custom_filter':'["cycleway:right"~"track"]', 'export': False, 'retain_all': True},
                   'bike_cyclewaylefttrack': {'network_type':'bike', 'custom_filter':'["cycleway:left"~"track"]', 'export': False, 'retain_all': True},
                   'bike_cyclestreet': {'network_type':'bike', 'custom_filter':'["cyclestreet"]', 'export': False, 'retain_all': True},
                   'bike_bicycleroad': {'network_type':'bike', 'custom_filter':'["bicycle_road"]', 'export': False, 'retain_all': True},
                   'bike_livingstreet': {'network_type':'bike', 'custom_filter':'["highway"~"living_street"]', 'export': False, 'retain_all': True}
                  }  
networktypes = ["biketrack", "carall", "bikeable", "biketrackcarall", "biketrack_onstreet", "bikeable_offstreet"] # Existing infrastructures to analyze

place_name = "Bucharest, Romania"
placeid = "Bucharest"
location = ox.geocoder.geocode_to_gdf(place_name)
location = fill_holes(extract_relevant_polygon(place_name, shapely.geometry.shape(location['geometry'][0])))

Gs = {}
for parameterid, parameterinfo in tqdm(osmnxparameters.items(), desc = "Networks", leave = False):
    for i in range(0,10): # retry
        try:
            Gs[parameterid] = ox.graph_from_polygon(location, 
                                   network_type = parameterinfo['network_type'],
                                   custom_filter = (parameterinfo['custom_filter']),
                                   retain_all = parameterinfo['retain_all'],
                                   simplify = False)
        except ValueError:
            Gs[parameterid] = nx.empty_graph(create_using = nx.MultiDiGraph)
            print(f"No OSM data for graph {parameterid}. Created empty graph.")
            break
        except ConnectionError or UnboundLocalError:
            print("ConnectionError or UnboundLocalError. Retrying.")
            continue
        except:
            print("Other error. Retrying.")
            continue
        break
    if parameterinfo['export']: ox_to_csv(Gs[parameterid], network_data_pth,placeid,parameterid)

# Compose special cases biketrack, bikeable, biketrackcarall
parameterid = 'biketrack'
Gs[parameterid] = nx.compose_all([Gs['bike_cyclewaylefttrack'], Gs['bike_cyclewaytrack'], Gs['bike_highwaycycleway'], Gs['bike_bicycleroad'], Gs['bike_cyclewayrighttrack'], Gs['bike_designatedpath'], Gs['bike_cyclestreet']])
ox_to_csv(Gs[parameterid], network_data_pth,placeid ,parameterid)
parameterid = 'bikeable'
Gs[parameterid] = nx.compose_all([Gs['biketrack'], Gs['car30'], Gs['bike_livingstreet']]) 
ox_to_csv(Gs[parameterid], network_data_pth, placeid, parameterid)
parameterid = 'biketrackcarall'
Gs[parameterid] = nx.compose(Gs['biketrack'], Gs['carall']) # Order is important
ox_to_csv(Gs[parameterid], network_data_pth, placeid, parameterid)


parameterid = 'toproutes'
top_streets = ["Bulevardul Lascăr Catargiu", "Bulevardul General Gheorghe Magheru", "Bulevardul Nicolae Bălcescu", "Bulevardul Brătianu", "Șoseaua Ștefan cel Mare", "Splaiul Unirii", "Bulevardul Unirii", "Șoseaua Mihai Bravu", "Bulevardul Iuliu Maniu",  "Splaiul Independenței", "Bulevardul Regina Elisabeta", "Bulevardul Mihail Kogălniceanu", "Drumul Taberei", "Calea Moșilor", "Șoseaua Colentina", "Strada Barbu Văcărescu", "Bulevardul Carol I"]
street_network = ox.graph_from_place(place_name, network_type="drive", retain_all=True)
G_filtered = nx.MultiDiGraph()
kept_nodes = set()
edges_kept = set()
for u, v, data in street_network.edges(data=True):
    # Check if 'name' exists and matches (OSMnx can have 'name' as a list sometimes)
    edge_name = data.get("name")
    if isinstance(edge_name, list):
        for name in edge_name:
            if name in top_streets:
                edges_kept.update([name])
                G_filtered.add_edge(u, v, **data)
                kept_nodes.update([u, v])
                break
    elif edge_name in top_streets:
        edges_kept.update([edge_name])
        G_filtered.add_edge(u, v, **data)
        kept_nodes.update([u, v])
print(edges_kept)
for node in kept_nodes:
    if node in street_network.nodes:
        G_filtered.add_node(node, **street_network.nodes[node])
Gs[parameterid] = G_filtered
ox_to_csv(Gs[parameterid], network_data_pth, placeid, parameterid)


for parameterid in networktypes[:-2]:
    #G_temp = nx.MultiDiGraph(ox.utils_graph.get_digraph(ox.simplify_graph(Gs[parameterid]))) # This doesnt work - cant get rid of multiedges
    ox_to_csv(ox.simplify_graph(Gs[parameterid]), network_data_pth,placeid ,parameterid, "_simplified")