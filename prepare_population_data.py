import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, box
from functions import project_nxpos, initplot, nx_to_ig, greedy_triangulation_routing, mst_routing, fill_holes, extract_relevant_polygon, osm_to_ig, nxdraw, ox_to_csv,csv_to_ox, csv_to_ig
import igraph as ig
import networkx as nx
import shapely
import numpy as np 
from matplotlib import cm
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path
from matplotlib.patches import Polygon as MplPolygon

data_pth = Path(".\Data")
network_data_pth = data_pth/"network_data"
poi_data_pth = data_pth/"poi_data"
population_data_pth = data_pth/"population_data"
population_density_file = population_data_pth/"population_density_ro.csv"

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

placeid = "Bucharest"
place_name = "Bucharest, Romania"
location = ox.geocoder.geocode_to_gdf(place_name)
location = fill_holes(extract_relevant_polygon(place_name, shapely.geometry.shape(location['geometry'][0])))
G_carall = csv_to_ox(network_data_pth, "Bucharest", "carall")
largest_cc = max(nx.weakly_connected_components(G_carall), key=len)
G_carall_largest = G_carall.subgraph(largest_cc).copy()
G_carall_largest.graph['crs'] = 'EPSG:4326'

pd_df = pd.read_csv(population_density_file)
pd_df.columns = ["x", "y", "density"]
grid_centers = pd_df.to_dict(orient="records")
grid_centers_gdf = gpd.GeoDataFrame(
    grid_centers, 
    geometry=[Point(d["x"], d["y"]) for d in grid_centers], 
    crs="EPSG:4326"
)
grid_centers_gdf["is_within"] = grid_centers_gdf.geometry.within(location)
bucharest_grid_centers = grid_centers_gdf[grid_centers_gdf['is_within'] == True]
bucharest_grid_centers_nnids = [ox.distance.nearest_nodes(G_carall_largest, row.geometry.x, row.geometry.y) for _, row in bucharest_grid_centers.iterrows()]
with open(population_data_pth/f'Bucharest_pdgridcenters_nnidscarall.csv', 'w') as f:
    for item in bucharest_grid_centers_nnids:
        f.write("%s\n" % item)
bucharest_grid_centers["nnid"] = bucharest_grid_centers_nnids
bucharest_grid_centers.to_csv(population_data_pth/"Bucharest_population_density_centers.csv")

#fig = initplot()
#ax = fig.gca()
#G_carall = csv_to_ig(network_data_pth, "Bucharest", "carall")
#map_center = nxdraw(G_carall, "carall", ax = ax)
#nxdraw(G_carall, "carall", map_center, ax = ax)

#G_nx = ox.simplify_graph(nx.MultiDiGraph(G_carall.to_networkx())).to_undirected()
#nnids_nx = [k for k,v in dict(G_nx.nodes(data=True)).items() if v['id'] in bucharest_grid_centers_nnids]
#G_temp = G_nx.subgraph(nnids_nx)
#bucharest_grid_centers_transformed, _  = project_nxpos(G_temp, map_center)
#squares = []
#densities = []
#for pt, density in zip(bucharest_grid_centers_transformed, bucharest_grid_centers.density):
#    x, y = bucharest_grid_centers_transformed[pt]
#    square = box(x - 500, y - 500, x + 500, y + 500)
#    squares.append(square)
#    densities.append(density)
#squares_gdf = gpd.GeoDataFrame(geometry=squares, crs="EPSG:32635").to_crs(epsg=4326)
#print(squares_gdf.crs)
#squares_gdf["density"] = densities
#for poly in squares_gdf.geometry:
#    square_patch = MplPolygon(
#        list(poly.exterior.coords),
#        closed=True,
#        edgecolor='red',
#        facecolor='skyblue',
#        alpha=0.5,
#        linewidth=2
#    )
#    ax.add_patch(square_patch)
#squares_gdf.plot(column='density', cmap='OrRd', legend=True, ax=ax)
#nxdraw(G_carall, "poi_unreached", map_center, bucharest_grid_centers_nnids, "nx.draw_networkx_nodes", 5)
#plt.savefig(population_data_pth/f'{placeid}_carall_poi.pdf', bbox_inches="tight", dpi=plotparam["dpi"])
