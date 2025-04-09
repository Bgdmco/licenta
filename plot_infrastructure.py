import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from functions import nx_to_ig, greedy_triangulation_routing, mst_routing, fill_holes, extract_relevant_polygon, osm_to_ig, nxdraw, ox_to_csv,csv_to_ox, csv_to_ig, write_result, nodesize_from_pois, initplot
import igraph as ig
import networkx as nx
import shapely
import numpy as np 
from matplotlib import cm
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path
import pickle

data_pth = Path(".\Data")
network_data_pth = data_pth/"network_data"
poi_data_pth = data_pth/"poi_data"
result_pth = data_pth/"results"
plots_pth = data_pth/"plots"

placeid = "Bucharest"
poi_source = "PlanBucuresti"


prune_measure = "betweenness"
prune_quantiles = [x/40 for x in list(range(1, 41))] # The quantiles where the GT should be pruned using the prune_measure

plotparam = {"bbox": (1280,1280),
			"dpi": 96,
			"carall": {"width": 0.5, "edge_color": '#999999'},
			# "biketrack": {"width": 1.25, "edge_color": '#2222ff'},
            "biketrack": {"width": 1, "edge_color": '#000000'},
			"biketrack_offstreet": {"width": 0.75, "edge_color": '#00aa22'},
			"bikeable": {"width": 0.75, "edge_color": '#222222'},
			# "bikegrown": {"width": 6.75, "edge_color": '#ff6200', "node_color": '#ff6200'},
			# "highlight_biketrack": {"width": 6.75, "edge_color": '#0eb6d2', "node_color": '#0eb6d2'},
            "bikegrown": {"width": 3.75, "edge_color": '#0eb6d2', "node_color": '#0eb6d2'},
            "highlight_biketrack": {"width": 3.75, "edge_color": '#2222ff', "node_color": '#2222ff'},
			"highlight_bikeable": {"width": 3.75, "edge_color": '#222222', "node_color": '#222222'},
			"poi_unreached": {"node_color": '#ff7338', "edgecolors": '#ffefe9'},
			"poi_reached": {"node_color": '#0b8fa6', "edgecolors": '#f1fbff'},
			"abstract": {"edge_color": '#000000', "alpha": 0.75}
			}
nodesize_grown = 7.5



if prune_measure == "betweenness":
    weight_abstract = True
else:
    weight_abstract = 6

# EXISTING INFRASTRUCTURE
# Load networks
G_carall = csv_to_ig(network_data_pth, placeid, "carall")
G_biketrackcarall = csv_to_ig(network_data_pth, placeid, 'biketrackcarall')
map_center = nxdraw(G_carall, "carall")

with open(poi_data_pth/f'{placeid}_poi_nnidscarall.csv') as f:
    nnids = [int(line.rstrip()) for line in f]

nodesize_poi = nodesize_from_pois(nnids)

# PLOT existing networks
fig = initplot()
nxdraw(G_carall, "carall", map_center)
plt.savefig(plots_pth/f'{placeid}_carall.pdf', bbox_inches="tight")
plt.savefig(plots_pth/f'{placeid}_carall.png', bbox_inches="tight", dpi=plotparam["dpi"])
plt.close()

G_toproutes = csv_to_ig(network_data_pth, placeid, 'toproutes')
G_biketrack = csv_to_ig(network_data_pth, placeid, 'biketrack')
print(set(G_toproutes.es['name']))
fig = initplot()
nxdraw(G_carall, "carall", map_center)
nxdraw(G_toproutes, "toproutes", map_center)
nxdraw(G_biketrack, "existing_biketrack", map_center)
plt.savefig(plots_pth/f'{placeid}_toproutes.pdf', bbox_inches="tight")
plt.savefig(plots_pth/f'{placeid}_toproutes.png', bbox_inches="tight", dpi=plotparam["dpi"])
plt.close()

try:
    fig = initplot()
    nxdraw(G_biketrack, "biketrack", map_center)
    plt.savefig(plots_pth/f'{placeid}_biketrack.pdf', bbox_inches="tight")
    plt.savefig(plots_pth/f'{placeid}_biketrack.png', bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()
    
    fig = initplot()
    nxdraw(G_carall, "carall", map_center)
    nxdraw(G_biketrack, "biketrack", map_center, list(set([v["id"] for v in G_biketrack.vs]).intersection(set([v["id"] for v in G_carall.vs]))))
    nxdraw(G_biketrack, "biketrack_offstreet", map_center, list(set([v["id"] for v in G_biketrack.vs]).difference(set([v["id"] for v in G_carall.vs]))))
    plt.savefig(plotparam/f'{placeid}_biketrackcarall.pdf', bbox_inches="tight")
    plt.savefig(plotparam/f'{placeid}_biketrackcarall.png', bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()
except:
    print(placeid + ": No bike tracks found")

try:
    G_bikeable = csv_to_ig(network_data_pth, placeid, 'bikeable')
    fig = initplot()
    nxdraw(G_bikeable, "bikeable", map_center)
    plt.savefig(plots_pth/f'{placeid}_bikeable.pdf', bbox_inches="tight")
    plt.savefig(plots_pth/f'{placeid}_bikeable.png', bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()
except:
    print(placeid + ": No bikeable infrastructure found")


fig = initplot()
nxdraw(G_carall, "carall", map_center)
nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
plt.savefig(plots_pth/f'{placeid}_carall_poi_{poi_source}.pdf', bbox_inches="tight")
plt.savefig(plots_pth/f'{placeid}_carall_poi_{poi_source}.png', bbox_inches="tight", dpi=plotparam["dpi"])
plt.close()