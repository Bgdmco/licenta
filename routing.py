import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from functions import nx_to_ig, greedy_triangulation_routing, mst_routing, fill_holes, extract_relevant_polygon, osm_to_ig, nxdraw, ox_to_csv,csv_to_ox, csv_to_ig, write_result
import igraph as ig
import networkx as nx
import shapely
import numpy as np 
from matplotlib import cm
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path

prune_measures = {"betweenness": "Bq", "closeness": "Cq", "random": "Rq"}
prune_quantiles = [x/40 for x in list(range(1, 41))] # The quantiles where the GT should be pruned using the prune_measure
prune_quantiles2 = [x/6 for x in list(range(1,7))]
data_pth = Path(".\Data")
network_data_pth = data_pth/"network_data"
poi_data_pth = data_pth/"poi_data"
result_pth = data_pth/"results"

G_carall = csv_to_ig(network_data_pth, "Bucharest", "carall")
with open(poi_data_pth/f'Bucharest_poi_nnidscarall.csv') as f:
    nnids = [int(line.rstrip()) for line in f]

(GTs, GT_abstracts) = greedy_triangulation_routing(G_carall, nnids, prune_quantiles2, "betweenness")
(MST, MST_abstract) = mst_routing(G_carall, nnids)

results = {"placeid": "Bucharest", "prune_measure": prune_measures["betweenness"], "poi_source": "PlanBucuresti", "prune_quantiles": prune_quantiles2, "GTs": GTs, "GT_abstracts": GT_abstracts, "MST": MST, "MST_abstract": MST_abstract}
write_result(result_pth,results, "pickle", "Bucharest", "PlanBucuresti_6prune_quantiles", prune_measures["betweenness"], ".pickle")