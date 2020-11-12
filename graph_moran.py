import time
t_import_0 = time.perf_counter() #to measure performance
import axelrod as axl
from axelrod.graph import Graph
import random
import matplotlib.pyplot as plt
from GraphGenerator import GraphGenerator
t_import_1 = time.perf_counter()
print(f"Imports took {t_import_1-t_import_0:0.5f} seconds")

def main():
    #randomly chosen list of strategies
    players = [axl.Cooperator(), axl.Defector(),
               axl.TitForTat(), axl.Grudger(),
               axl.Alternator(), axl.AdaptorBrief(),
               axl.AdaptorLong(), axl.Random(),
               axl.Retaliate(), axl.Punisher()]
    #initiate graph for visualisation and automatic edge production
    gen_graph = GraphGenerator(players)
    '''
    Now create the edges for axelrod
    GraphGenerator will do this automatically for some graph types
    but a properly formatted list may also be supplied
    if a specific structure is desired that cannot be
    automatically generated.
    If the structure may be automatically generated,
    please write it in the GraphGenerator class
    '''
    edges = gen_graph.ring_graph()
    axl_graph = Graph(edges) #makes axelrod graph instant, for the actual game
    mp = axl.MoranProcess(players, interaction_graph=axl_graph, seed=1)
    populations = mp.play()
    print(f"There were {len(mp)} rounds in the simple Moran game")
    print(f"{mp.winning_strategy_name} won the simple Moran game")
    mp.populations_plot()
    gen_graph.visualise_graph(edges)

    #to measure performance, help us determine if we will need any HPC time
    t1 = time.perf_counter()
    print(f"Simple Moran program ran in {t1-t_import_1:0.5f} seconds")

    plt.show() #finally show all plots called
main()

def mutation():
    t2 = time.perf_counter()
    #randomly chosen list of strategies
    players = [axl.Cooperator(), axl.Defector(),
               axl.TitForTat(), axl.Grudger(),
               axl.Alternator(), axl.AdaptorBrief(),
               axl.AdaptorLong(), axl.Random(),
               axl.Retaliate(), axl.Punisher()]
    #initiate graph for visualisation and automatic edge production
    gen_graph = GraphGenerator(players)
    '''
    Now create the edges for axelrod
    GraphGenerator will do this automatically for some graph types
    but a properly formatted list may also be supplied
    if a specific structure is desired that cannot be
    automatically generated.
    If the structure may be automatically generated,
    please write it in the GraphGenerator class
    '''
    edges = gen_graph.ring_graph()
    axl_graph = Graph(edges) #makes axelrod graph instant, for the actual game

    mp = axl.MoranProcess(
    players, interaction_graph=axl_graph, mutation_rate=0.05, seed=10
    )
    #this loop stops the game running indefinitely
    for _ in mp:
        if len(mp.population_distribution()) == 1:
            break
    mp.population_distribution() #run the game
    mp.populations_plot()
    gen_graph.visualise_graph(edges)

    t3 = time.perf_counter()
    print(f"Mutation program ran in {t3-t2:0.5f} seconds")

    plt.show()
mutation()
