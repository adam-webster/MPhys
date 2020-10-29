import time
t_import_0 = time.perf_counter() #to measure performance
import axelrod as axl
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pprint
t_import_1 = time.perf_counter()


def main():

    #random list of strategies chosen
    players = [axl.Cooperator(), axl.Defector(),
               axl.TitForTat(), axl.Grudger(),
               axl.Alternator(), axl.AdaptorBrief(),
               axl.AdaptorLong()]

    tournament = axl.Tournament(players)
    results = tournament.play()
    #print(results.ranked_names)
    #pprint.pprint(results.summarise())
    plot = axl.Plot(results)
    #can just do axl.Plot(results).boxplot().show()
    #this next code block gives better formatting options
    fig, ax = plt.subplots()
    title = ax.set_title('Payoff')
    xlabel = ax.set_xlabel('Strategies')
    plot.boxplot(ax=ax)

    #other results visuals available
    plot.payoff()


    #or save all available to file, doesn't seem to work though
    #plot.save_all_plots()

    #to measure performance, help us determine if we will need any HPC time
    t1 = time.perf_counter()
    print(f"Imports took {t_import_1-t_import_0:0.5f} seconds")
    print(f"Program ran in {t1-t_import_1:0.5f} seconds")

    plt.show() #to show all plots called
main()
