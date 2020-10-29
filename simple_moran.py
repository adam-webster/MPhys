import time
t_import_0 = time.perf_counter() #to measure performance
import axelrod as axl
import random, pprint
import matplotlib.pyplot as plt
t_import_1 = time.perf_counter()


def main():
    #random list of strategies chosen
    players = [axl.Cooperator(), axl.Defector(),
               axl.TitForTat(), axl.Grudger(),
               axl.Alternator(), axl.AdaptorBrief(),
               axl.AdaptorLong(), axl.Random(),
               axl.Random(), axl.Random()]

    mp = axl.MoranProcess(players, seed=1)
    populations = mp.play()
    print(len(mp))
    print(mp.winning_strategy_name)

    mp.populations_plot()


    #to measure performance, help us determine if we will need any HPC time
    t1 = time.perf_counter()
    print(f"Imports took {t_import_1-t_import_0:0.5f} seconds")
    print(f"Simple Moran program ran in {t1-t_import_1:0.5f} seconds")

    plt.show() #finally show all plots called
main()

def mutation():
    t2 = time.perf_counter()
    #random list of strategies chosen
    players = [axl.Cooperator(), axl.Defector(),
               axl.TitForTat(), axl.Grudger(),
               axl.Alternator(), axl.AdaptorBrief(),
               axl.AdaptorLong(), axl.Random(),
               axl.Appeaser(), axl.CooperatorHunter()]


    mp = axl.MoranProcess(players, mutation_rate=0.3)
    for _ in mp:
        if len(mp.population_distribution()) == 1:
            break
    mp.population_distribution()
    print(mp.winning_strategy_name)
    mp.populations_plot()

    t3 = time.perf_counter()
    print(f"Mutation program ran in {t3-t2:0.5f} seconds")

    plt.show()
mutation()
