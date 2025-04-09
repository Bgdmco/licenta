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
if prune_measure == "betweenness":
    weight_abstract = True
else:
    weight_abstract = 6

#PLOT_PARAMETERS
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



G_carall = csv_to_ig(network_data_pth, "Bucharest", "carall")
map_center = nxdraw(G_carall, "carall")

with open(poi_data_pth/f'Bucharest_poi_nnidscarall.csv') as f:
    nnids = [int(line.rstrip()) for line in f]

nodesize_poi = nodesize_from_pois(nnids)
fig = initplot()
nxdraw(G_carall, "carall", map_center)
nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
plt.savefig(plots_pth/f"{placeid}_carall_poi_{poi_source}.pdf", bbox_inches="tight")
plt.savefig(plots_pth/f"{placeid}_carall_poi_{poi_source}.png", bbox_inches="tight", dpi=plotparam["dpi"])
plt.close()


with open(result_pth/"Bucharest_poi_PlanBucuresti_Bq.pickle", 'rb') as f:
    res = pickle.load(f)

# PLOT abstract MST
fig = initplot()
nxdraw(G_carall, "carall", map_center)
nxdraw(res["MST_abstract"], "abstract", map_center, weighted = 6)
nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in res["MST"].vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
plt.savefig(plots_pth/f"{placeid}_MSTabstract_poi_{poi_source}.pdf", bbox_inches="tight")
plt.savefig(plots_pth/f"{placeid}_MSTabstract_poi_{poi_source}.png", bbox_inches="tight", dpi=plotparam["dpi"])
plt.close()

# PLOT MST all together
fig = initplot()
nxdraw(G_carall, "carall")
nxdraw(res["MST"], "bikegrown", map_center, nodesize = nodesize_grown)
nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in res["MST"].vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
plt.savefig(plots_pth/f"{placeid}_MSTall_poi_{poi_source}.pdf", bbox_inches="tight")
plt.savefig(plots_pth/f"{placeid}_MSTall_poi_{poi_source}.png", bbox_inches="tight", dpi=plotparam["dpi"])
plt.close()

# PLOT MST all together with abstract
fig = initplot()
nxdraw(G_carall, "carall", map_center)
nxdraw(res["MST"], "bikegrown", map_center, nodesize = 0)
nxdraw(res["MST_abstract"], "abstract", map_center, weighted = 6)
nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in res["MST"].vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
plt.savefig(plots_pth/f"{placeid}_MSTabstractall_poi_{poi_source}.pdf", bbox_inches="tight")
plt.savefig(plots_pth/f"{placeid}_MSTabstractall_poi_{poi_source}.png", bbox_inches="tight", dpi=plotparam["dpi"])
plt.close()

# PLOT abstract greedy triangulation (this can take some minutes)
for GT_abstract, prune_quantile in tqdm(zip(res["GT_abstracts"], res["prune_quantiles"]), "Abstract triangulation", total=len(res["prune_quantiles"])):
    fig = initplot()
    nxdraw(G_carall, "carall")
    try:
        GT_abstract.es["weight"] = GT_abstract.es["width"]
    except:
        pass
    nxdraw(GT_abstract, "abstract", map_center, drawfunc = "nx.draw_networkx_edges", nodesize = 0, weighted = weight_abstract, maxwidthsquared = nodesize_poi)
    nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
    nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in GT_abstract.vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
    plt.savefig(plots_pth/f"{placeid}_GTabstract_poi_{poi_source}_{prune_measure}{prune_quantile:.3f}.png", bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()

# PLOT all together (this can take some minutes)
for GT, prune_quantile in tqdm(zip(res["GTs"], res["prune_quantiles"]), "Routed triangulation", total=len(res["prune_quantiles"])):
    fig = initplot()
    nxdraw(G_carall, "carall")
    nxdraw(GT, "bikegrown", map_center, nodesize = nodesize_grown)
    nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
    nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in GT.vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
    plt.savefig(plots_pth/f"{placeid}_GTall_poi_{poi_source}_{prune_measure}{prune_quantile:.3f}.png", bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()
     