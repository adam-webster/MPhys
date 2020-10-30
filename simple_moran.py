import time
t_import_0 = time.perf_counter() #to measure performance
import axelrod as axl
import random, pprint
import matplotlib.pyplot as plt
t_import_1 = time.perf_counter()
print(f"Imports took {t_import_1-t_import_0:0.5f} seconds")

def main():
    #random list of strategies chosen
    players = [axl.Cooperator(), axl.Defector(),
               axl.TitForTat(), axl.Grudger(),
               axl.Alternator(), axl.AdaptorBrief(),
               axl.AdaptorLong(), axl.Random(),
               axl.Retaliate(), axl.Punisher()]

    mp = axl.MoranProcess(players, seed=1)
    populations = mp.play()
    print(f"There were {len(mp)} rounds in the simple Moran game")
    print(f"{mp.winning_strategy_name} won the simple Moran game")
    mp.populations_plot()

    #to measure performance, help us determine if we will need any HPC time
    t1 = time.perf_counter()
    print(f"Simple Moran program ran in {t1-t_import_1:0.5f} seconds")

    plt.show() #finally show all plots called
main()

def mutation():
    t2 = time.perf_counter()
    #random list of strategies chosen
    '''
    players = [axl.Cooperator(), axl.Defector(),
               axl.TitForTat(), axl.Grudger(),
               axl.Alternator(), axl.AdaptorBrief(),
               axl.AdaptorLong(), axl.Random(),
               axl.Appeaser(), axl.CooperatorHunter()]
               '''
    players = [axl.Cooperator(), axl.Defector(),
               axl.TitForTat(), axl.Grudger(),
               axl.Alternator(), axl.AdaptorBrief(),
               axl.AdaptorLong(), axl.Random(),
               axl.Retaliate(), axl.Punisher()]


    mp = axl.MoranProcess(players, mutation_rate=0.1, seed=10)
    #this loop stops the game running indefinitely
    for _ in mp:
        if len(mp.population_distribution()) == 1:
            break
    mp.population_distribution() #run the game
    print(f"{mp.winning_strategy_name} won the mutation enabled game")
    mp.populations_plot()

    t3 = time.perf_counter()
    print(f"Mutation program ran in {t3-t2:0.5f} seconds")

    plt.show()
mutation()
