import time
t_import_0 = time.perf_counter() #to measure performance
import axelrod as axl
import matplotlib.pyplot as plt
from GraphGenerator import GraphGenerator
t_import_1 = time.perf_counter()
print(f"Imports took {t_import_1-t_import_0:0.5f} seconds")


def main():
    #randomly chosen list of strategies
    players = [axl.Cooperator(), axl.Defector(),
               axl.TitForTat(), axl.Grudger(),
               axl.Alternator(), axl.AdaptorBrief(),
               axl.AdaptorLong(), axl.Adaptive()]
    gen_graph = GraphGenerator(players) #creates the graph object
    #create the edges for axelrod
    #GraphGenerator will do this automatically for some graph types
    #but a properly formatted list may also be supplied
    edges = gen_graph.ring_graph()
    spatial_tournament = axl.Tournament(players, edges=edges)
    results = spatial_tournament.play()

    #now for visualisations and graphs
    gen_graph.visualise_graph(edges)
    plot = axl.Plot(results)
    #this next code block gives better formatting options
    fig, ax = plt.subplots()
    title = ax.set_title('Payoff')
    xlabel = ax.set_xlabel('Strategies')
    plot.boxplot(ax=ax)

    #other results visuals available
    plot.payoff()


    #to measure performance, help us determine if we will need any HPC time
    t1 = time.perf_counter()
    print(f"Program ran in {t1-t_import_1:0.5f} seconds")

    plt.show() #to show all plots called
main()
