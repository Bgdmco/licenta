
import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from functions import write_result, calculate_metrics, csv_to_ig, calculate_metrics_additively, set_analysissubplot, create_pop_density_proj, initplot, nxdraw
import igraph as ig
import networkx as nx
import shapely
import numpy as np 
from matplotlib import cm
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path
from matplotlib.patches import Polygon as MplPolygon
import copy
import os
import itertools
import pickle
from matplotlib.ticker import MaxNLocator

def analyse_existing_infrastructure(Gs, G_population_centers,nnids, analysis_results_pth, placeid):
    print(placeid + ": Analyzing existing infrastructure... (this can take several minutes)")
    population_squares = create_pop_density_proj(G_carall, G_population_centers, 500)
    empty_metrics = {
                     "length":0,
                     "length_lcc":0,
                     "coverage": 0,
                     "directness": 0,
                     "directness_lcc": 0,
                     "poi_coverage": 0,
                     "components": 0,
                     "efficiency_global": 0,
                     "efficiency_local": 0,
                     "efficiency_global_routed": 0,
                     "efficiency_local_routed": 0,
                     "directness_lcc_linkwise": 0,
                     "directness_all_linkwise": 0,
                     "population_coverage" : 0
                    }
    output_place = {}
    for networktype in networktypes:
        output_place[networktype] = copy.deepcopy(empty_metrics)
    # Analyze all networks     

    covs = {}
    for networktype in tqdm(networktypes, desc = "Networks", leave = False):
        try:
            metrics, cov = calculate_metrics(Gs[networktype], Gs[networktype + "_simplified"], Gs['carall'],population_squares,nnids, empty_metrics)
            for key, val in metrics.items():
                output_place[networktype][key] = val
            covs[networktype] = cov
        except Exception as e:
            print(networktype + " is empty")
            raise(e)

    # Save the covers
    write_result(analysis_results_pth,covs, "pickle", placeid, "", "", "existing_covers.pickle")

    # Write to CSV
    write_result(analysis_results_pth, output_place, "dictnested", placeid, "", "", "existing.csv", empty_metrics)

def analyse_poi_based_results(res, nnids, Gexisting, G_carall, G_population_centers,analysis_results_pth, placeid, poi_source):
    # Calculate
    # output contains lists for all the prune_quantile values of the corresponding results
    population_squares = create_pop_density_proj(G_carall, G_population_centers, 500)
    output, covs = calculate_metrics_additively(res["GTs"], res["GT_abstracts"],population_squares,res["prune_quantiles"], G_carall, nnids, return_cov = True, Gexisting=Gexisting)
    output_MST, cov_MST = calculate_metrics(res["MST"], res["MST_abstract"], G_carall,population_squares,nnids, output, return_cov=True, Gexisting = Gexisting)

    # Save the covers
    write_result(analysis_results_pth, covs, "pickle", placeid, poi_source, "Bq", "_covers.pickle")
    #    write_result(covs_carminusbike, "pickle", placeid, poi_source, prune_measure, "_covers_carminusbike.pickle")
    write_result(analysis_results_pth, cov_MST, "pickle", placeid, poi_source, "Bq", "_cover_mst.pickle")

    # Write to CSV
    write_result(analysis_results_pth, output, "dict", placeid, poi_source, "Bq", ".csv")
    #    write_result(output_carminusbike, "dict", placeid, poi_source, prune_measure, "_carminusbike.csv")
    #    write_result(output_carconstrictedbike, "dict", placeid, poi_source, prune_measure, "_carconstrictedbike.csv")
    write_result(analysis_results_pth, output_MST, "dict", placeid, poi_source, "", "mst.csv")

def plot_analysis(poi_source_list, prune_measure_list, prune_quantiles,analysis_results_pth, plotconstricted = False):
    analysis_existing_rowkeys = {"bikeable": 0, "bikeable_offstreet": 1, "biketrack": 2, "biketrack_onstreet": 3, "biketrackcarall": 4, "carall": 5}
    plotparam_analysis = {
			"bikegrown": {"linewidth": 3.75, "color": '#0eb6d2', "linestyle": "solid", "label": "Grown network"},
			"bikegrown_abstract": {"linewidth": 3.75, "color": '#000000', "linestyle": "solid", "label": "Grown network (unrouted)", "alpha": 0.75},
			"mst": {"linewidth": 2, "color": '#0eb6d2', "linestyle": "dashed", "label": "MST"},
			"mst_abstract": {"linewidth": 2, "color": '#000000', "linestyle": "dashed", "label": "MST (unrouted)", "alpha": 0.75},
			"biketrack": {"linewidth": 1, "color": '#2222ff', "linestyle": "solid", "label": "Protected"},
			"bikeable": {"linewidth": 1, "color": '#222222', "linestyle": "dashed", "label": "Bikeable"},
			"constricted": {"linewidth": 3.75, "color": '#D22A0E', "linestyle": "solid", "label": "Street network"},
            "constricted_SI": {"linewidth": 2, "color": '#D22A0E', "linestyle": "solid", "label": "Street network"},
			"constricted_3": {"linewidth": 2, "color": '#D22A0E', "linestyle": "solid", "label": "Top 3%"},
			"constricted_5": {"linewidth": 2, "color": '#a3210b', "linestyle": "solid", "label": "Top 5%"},
			"constricted_10": {"linewidth": 2, "color": '#5a1206', "linestyle": "solid", "label": "Top 10%"},
            "bikegrown_betweenness": {"linewidth": 2.5, "color": '#0eb6d2', "linestyle": "solid", "label": "Betweenness"},
            "bikegrown_closeness": {"linewidth": 2, "color": '#186C7A', "linestyle": "dashed", "label": "Closeness"},
            "bikegrown_random": {"linewidth": 1.5, "color": '#222222', "linestyle": "dotted", "label": "Random"}
			}
    # Either: Run all parameter sets
    # poi_source_list = ["grid", "railwaystation"]
    # prune_measure_list = ["betweenness", "closeness", "random"]
    parsets_used = list(itertools.product(poi_source_list, prune_measure_list))


    for poi_source_this, prune_measure_this in parsets_used:
        print(poi_source_this, prune_measure_this)



        # PLOT Analysis
        filename = f"{placeid}_poi_{poi_source_this}_{prune_measure_this}.csv"
        analysis_result = np.genfromtxt(analysis_results_pth/filename, delimiter=',', names=True)
        if len(analysis_result) == 0: # No plot if no results (for example no railwaystations)
            print(placeid + ": No analysis results available")
            continue
        print(placeid + ": Plotting analysis results...")
        # Load existing networks
        # G_biketrack = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'biketrack')
        # G_carall = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'carall')
        # G_bikeable = csv_to_ig(PATH["data"] + placeid + "/", placeid, 'bikeable')
        #    G_biketrack_onstreet = intersect_igraphs(G_biketrack, G_carall)
        #    G_bikeable_onstreet = intersect_igraphs(G_bikeable, G_carall)
        filename = f"{placeid}_poi_{poi_source_this}_mst.csv"
        analysis_mst_result = np.genfromtxt(analysis_results_pth/filename, delimiter=',', names=True)
        filename = f"{placeid}_existing.csv"
        analysis_existing = np.genfromtxt(analysis_results_pth/filename, delimiter=',', names=True)
        prune_quantiles_constricted = prune_quantiles
        if plotconstricted:
            f = f"results_constricted" + "results_" + poi_source_this + "_" + prune_measure_this + "/metrics_" + poi_source_this + "_" + prune_measure_this + "/" + placeid + "_carconstrictedbike_poi_" + poi_source_this + "_" + prune_measure_this + ".csv"
            if os.path.isfile(f):
                analysis_result_constricted = np.loadtxt(f, delimiter=',', usecols = (2,3,4,5,6,7,8,9,10), skiprows=1)
                if np.shape(analysis_result_constricted)[0] == 3: # for large cities we only calculated 3 values
                    prune_quantiles_constricted = [prune_quantiles[19], prune_quantiles[-1]]
        nc = 5
        fig, axes = plt.subplots(nrows = 2, ncols = nc, figsize = (16, 6))
        # Bike network
        keys_metrics = {"length": "Length [km]","coverage": "Coverage [km$^2$]","overlap_biketrack": "Overlap Existing Network","directness_all_linkwise": "Directness","efficiency_global": "Global Efficiency",
                "length_lcc": "Length of LCC [km]","poi_coverage": "POIs covered","components": "Components","efficiency_local": "Local Efficiency", "population_coverage" : "Population Covered"}
        for i, ax in enumerate(axes[0]):
            key = list(keys_metrics.keys())[i]
            if key in ["overlap_biketrack", "overlap_bikeable"]:
                ax.plot(prune_quantiles, analysis_result[key] / analysis_result["length"], **plotparam_analysis["bikegrown"])
                xmin, xmax = ax.get_xlim()
                ax.plot([xmin, xmax], [analysis_mst_result[key]/analysis_mst_result['length'], analysis_mst_result[key]/analysis_mst_result['length']], **plotparam_analysis["mst"])
            elif key in ["efficiency_global", "efficiency_local"]:
                ax.plot(prune_quantiles, analysis_result[key], **plotparam_analysis["bikegrown_abstract"])
                xmin, xmax = ax.get_xlim()
                tmp, = ax.plot([xmin, xmax], [analysis_mst_result[key], analysis_mst_result[key]], **plotparam_analysis["mst"])  # MST is equivalent for abstract and routed
                tmp.set_label('_hidden')
                tmp, = ax.plot(prune_quantiles, analysis_result[key+"_routed"], **plotparam_analysis["bikegrown"])
                tmp.set_label('_hidden')
            elif key in ["length", "length_lcc"]: # Convert m->km
                ax.plot(prune_quantiles, analysis_result[key]/1000, **plotparam_analysis["bikegrown"])
                xmin, xmax = ax.get_xlim()
                ax.plot([xmin, xmax], [analysis_mst_result[key]/1000, analysis_mst_result[key]/1000], **plotparam_analysis["mst"])
            else:
                ax.plot(prune_quantiles, analysis_result[key], **plotparam_analysis["bikegrown"])
                xmin, xmax = ax.get_xlim()
                ax.plot([xmin, xmax], [analysis_mst_result[key], analysis_mst_result[key]], **plotparam_analysis["mst"])
            try:
                if key in ["length", "length_lcc"]: # Convert m->km
                    tmp, = ax.plot([xmin, xmax], [analysis_existing[key][analysis_existing_rowkeys["biketrack"]]/1000, analysis_existing[key][analysis_existing_rowkeys["biketrack"]]/1000], **plotparam_analysis["biketrack"])
                else:
                    tmp, = ax.plot([xmin, xmax], [analysis_existing[key][analysis_existing_rowkeys["biketrack"]], analysis_existing[key][analysis_existing_rowkeys["biketrack"]]], **plotparam_analysis["biketrack"])
                if key in ["efficiency_global", "efficiency_local"]:
                    tmp.set_label('_hidden')
                if key in ["length", "length_lcc"]: # Convert m->km
                    tmp, = ax.plot([xmin, xmax], [analysis_existing[key][analysis_existing_rowkeys["bikeable"]]/1000, analysis_existing[key][analysis_existing_rowkeys["bikeable"]]/1000], **plotparam_analysis["bikeable"])
                else:
                    tmp, = ax.plot([xmin, xmax], [analysis_existing[key][analysis_existing_rowkeys["bikeable"]], analysis_existing[key][analysis_existing_rowkeys["bikeable"]]], **plotparam_analysis["bikeable"])
                if key in ["efficiency_global", "efficiency_local"]:
                    tmp.set_label('_hidden')
            except:
                pass
            if key == "efficiency_global" and plotconstricted:
                ax.plot([0]+prune_quantiles_constricted, analysis_result_constricted[:, 0], **plotparam_analysis["constricted"])
            if i == 0:
                ymax0 = ax.get_ylim()[1]
                ax.set_ylim(0, ymax0)
                ax.text(-0.15, ymax0*1.25, "Bucharest" + " (" + poi_source_this + " | " + prune_measure_this + ")", fontsize=16, horizontalalignment='left')
                ax.legend(loc = 'upper left')
            if i == 4:
                ax.legend(loc = 'best')
            if key == "directness_all_linkwise" and plotconstricted:
                ax.plot([0]+prune_quantiles_constricted, analysis_result_constricted[:, -1], **plotparam_analysis["constricted"])
            set_analysissubplot(key, ax)
            ax.set_title(list(keys_metrics.values())[i])
            ax.set_xlabel('')
            ax.set_xticklabels([])
        for i, ax in enumerate(axes[1]):
            key = list(keys_metrics.keys())[i+nc]
            if key == "population_coverage":
                ax.plot(prune_quantiles, analysis_result[key], **plotparam_analysis["bikegrown"])
                xmin, xmax = ax.get_xlim()
                ax.plot([xmin, xmax], [analysis_mst_result[key], analysis_mst_result[key]], **plotparam_analysis["mst"])
            elif key in ["efficiency_global", "efficiency_local"]:
                ax.plot(prune_quantiles, analysis_result[key], **plotparam_analysis["bikegrown_abstract"])
                xmin, xmax = ax.get_xlim()
                ax.plot([xmin, xmax], [analysis_mst_result[key], analysis_mst_result[key]], **plotparam_analysis["mst"]) # MST is equivalent for abstract and routed
                ax.plot(prune_quantiles, analysis_result[key+"_routed"], **plotparam_analysis["bikegrown"])
            elif key in ["length", "length_lcc"]: # Convert m->km
                ax.plot(prune_quantiles, analysis_result[key]/1000, **plotparam_analysis["bikegrown"])
                xmin, xmax = ax.get_xlim()
                ax.plot([xmin, xmax], [analysis_mst_result[key]/1000, analysis_mst_result[key]/1000], **plotparam_analysis["mst"])
            else:
                ax.plot(prune_quantiles, analysis_result[key], **plotparam_analysis["bikegrown"])
                xmin, xmax = ax.get_xlim()
                ax.plot([xmin, xmax], [analysis_mst_result[key], analysis_mst_result[key]], **plotparam_analysis["mst"])
            try:
                if key in ["length", "length_lcc"]: # Convert m->km
                    ax.plot([xmin, xmax], [analysis_existing[key][analysis_existing_rowkeys["biketrack"]]/1000, analysis_existing[key][analysis_existing_rowkeys["biketrack"]]/1000], **plotparam_analysis["biketrack"])
                    ax.plot([xmin, xmax], [analysis_existing[key][analysis_existing_rowkeys["bikeable"]]/1000, analysis_existing[key][analysis_existing_rowkeys["bikeable"]]/1000], **plotparam_analysis["bikeable"])
                else:
                    if not (key == "poi_coverage" and poi_source_this == "railwaystation"):
                        ax.plot([xmin, xmax], [analysis_existing[key][analysis_existing_rowkeys["biketrack"]], analysis_existing[key][analysis_existing_rowkeys["biketrack"]]], **plotparam_analysis["biketrack"])
                        ax.plot([xmin, xmax], [analysis_existing[key][analysis_existing_rowkeys["bikeable"]], analysis_existing[key][analysis_existing_rowkeys["bikeable"]]], **plotparam_analysis["bikeable"])
            except:
                pass
            if key == "efficiency_local" and plotconstricted:
                ax.plot([0]+prune_quantiles_constricted, analysis_result_constricted[:, 1], **plotparam_analysis["constricted"])
            if i == 0:
                ax.set_ylim(0, ymax0)
            set_analysissubplot(key, ax)
            ax.set_title(list(keys_metrics.values())[i+nc])
            ax.set_xlabel(prune_measure_this + ' quantile')
            if key in ["poi_coverage"]:
                # https://stackoverflow.com/questions/30914462/matplotlib-how-to-force-integer-tick-labels
                ax.yaxis.set_major_locator(MaxNLocator(integer=True)) 
            plt.subplots_adjust(top = 0.87, bottom = 0.09, left = 0.05, right = 0.97, wspace = 0.25, hspace = 0.25)
            if plotconstricted:
                fig.savefig(analysis_results_pth + placeid + "/" + placeid + '_analysis_poi_' + poi_source_this + "_" + prune_measure_this + '.png', facecolor = "white", edgecolor = 'none')
            else:
                fig.savefig(analysis_results_pth/f"{placeid}_analysis_poi_{poi_source_this}_{prune_measure_this}_noconstr.png", facecolor = "white", edgecolor = 'none')
            plt.close()

if __name__ == "__main__":
    placeid = "Bucharest"
    place_name = "Bucharest, Romania"
    poi_source = "PlanBucuresti"
    networktypes = ["carall", "bikenetwork"] # Existing infrastructures to analyze
    prune_quantiles = [x/40 for x in list(range(1, 41))] # The quantiles where the GT should be pruned using the prune_measure

    data_pth = Path(".\Data")
    network_data_pth = data_pth/"network_data"
    poi_data_pth = data_pth/"poi_data"
    analysis_results_pth = data_pth/"analysis_results"
    result_pth = data_pth/"results"
    population_data_pth = data_pth/"population_data"
    with open(poi_data_pth/f'{placeid}_poi_nnidscarall.csv') as f:
        nnids = [int(line.rstrip()) for line in f]
    G_toproutes = csv_to_ig(network_data_pth, placeid, 'toproutes')
    G_biketrack = csv_to_ig(network_data_pth, placeid, 'biketrack')
    G_toproutes_simplified = csv_to_ig(network_data_pth, placeid, 'toproutes_simplified')
    G_biketrack_simplified = csv_to_ig(network_data_pth, placeid, 'biketrack_simplified')

    G_population_centers = csv_to_ig(population_data_pth, placeid, 'population_density_centers')
    G_carall = csv_to_ig(network_data_pth, placeid, "carall")
    G_carall_simplified = csv_to_ig(network_data_pth, placeid, "carall_simplified")
    G_bikenetwork = G_biketrack + G_toproutes
    G_bikenetwork_simplified = G_biketrack_simplified + G_toproutes_simplified
    Gs = {}
    for networktype in networktypes:
        Gs[networktype] = eval(f"G_{networktype}")
        Gs[f"{networktype}_simplified"] = eval(f"G_{networktype}_simplified")
    # output_place is one static file for the existing city. This can be compared to the generated infrastructure.
    # Make a check if this file was already generated - it only needs to be done once. If not, generate it:
    filename = f"{placeid}_existing.csv"
    analyze_existing = False
    if not os.path.isfile(analysis_results_pth/filename) or analyze_existing:
        analyse_existing_infrastructure(Gs,G_population_centers, nnids, analysis_results_pth, placeid)
    
    with open(result_pth/"Bucharest_poi_PlanBucuresti_Bq.pickle", 'rb') as f:
        res = pickle.load(f)
    Gexisting = {}
    Gexisting["biketrack"] = G_bikenetwork
    poi_based_analysis = False
    population_squares = create_pop_density_proj(G_carall, G_population_centers, 500)
    if poi_based_analysis:
        analyse_poi_based_results(res, nnids, Gexisting,G_carall, G_population_centers,analysis_results_pth, placeid, poi_source)
    plot_analysis([poi_source], ["Bq"], prune_quantiles, analysis_results_pth)