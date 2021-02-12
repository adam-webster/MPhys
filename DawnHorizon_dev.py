"""
Author: Adam W
This will produce new, untested simulations of
evolutionary games on complex networks
"""

import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from progress.bar import IncrementalBar

#to format the help screen nicely
class CustomHelpFormatter(argparse.HelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ', '.join(action.option_strings) + ' ' + args_string

    def __init__(self, prog):
        super().__init__(prog, max_help_position=40, width=80)

COOP = 1
DEFEC = 0
vals = [COOP, DEFEC]

def randomGrid(N):
    """returns a grid of NxN random values.
       Probability of cooperator is 0.9"""
    return np.random.choice(vals, N*N, p=[0.9, 0.1]).reshape(N, N)

def addGlider(i, j, grid):
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 0],
                       [1, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0]])
    grid[i:i+4, j:j+5] = glider

def addRotator(i, j, grid):
    '''Adds a rotator (as defined in Nowak & May)
       with top left cell at (i, j)'''
    rotator = np.array([[1, 1],
                       [1, 1],
                       [1, 0],
                       [1, 0]])
    grid[i:i+4, j:j+2] = rotator

def addGrower(i, j, grid):
    '''Adds a grower (as defined in Nowak & May)
       with top left cell at (i, j)'''
    grower = np.array([[1, 1, 1, 1],
                       [0, 0, 1, 1],
                       [0, 0, 1, 1],
                       [0, 0, 1, 1]])
    grid[i:i+4, j:j+4] = grower

def kings_neighbours(N, i, j, grid, scoreGrid, scoring=False):
    '''This will generate a list of the 8 nearest neighbours,
       in kings move format, to location (i, j) on the grid.
       Boundary conditions are toroidal.
       Scoring parameter to be declared if you want to access the
       grid of scores'''
    if not scoring:
        neighbours = [grid[i, (j-1)%N], grid[i, (j+1)%N],
           grid[(i-1)%N, j], grid[(i+1)%N, j],
           grid[(i-1)%N, (j-1)%N], grid[(i-1)%N, (j+1)%N],
           grid[(i+1)%N, (j-1)%N], grid[(i+1)%N, (j+1)%N]]
    if scoring:
        neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[i, (j+1)%N],
           scoreGrid[(i-1)%N, j], scoreGrid[(i+1)%N, j],
           scoreGrid[(i-1)%N, (j-1)%N], scoreGrid[(i-1)%N, (j+1)%N],
           scoreGrid[(i+1)%N, (j-1)%N], scoreGrid[(i+1)%N, (j+1)%N]]
    return neighbours


def fixed_bc_kings_neighbours(N, i, j, grid, scoreGrid, scoring=False):
    '''This will generate a list of nearest 8 neighbours,
       in kings move format, to location (i, j) on the grid.
       Boundary conditions are now fixed.
       Scoring parameter to be declared if you want to access the
       grid of scores'''
    if not scoring:
        if i >= 1 and i < N-1:
            if j >= 1 and j < N-1:
                neighbours = [grid[i, (j-1)%N], grid[i, (j+1)%N],
                    grid[(i-1)%N, j], grid[(i+1)%N, j],
                    grid[(i-1)%N, (j-1)%N], grid[(i-1)%N, (j+1)%N],
                    grid[(i+1)%N, (j-1)%N], grid[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [grid[i, (j+1)%N], grid[(i-1)%N, j],
                    grid[(i+1)%N, j], grid[(i-1)%N, (j+1)%N],
                    grid[(i+1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [grid[i, (j-1)%N], grid[(i-1)%N, j],
                    grid[(i+1)%N, j], grid[(i-1)%N, (j-1)%N],
                    grid[(i+1)%N, (j-1)%N]]
        elif i == 0:
            if j >= 1 and j < N-1:
                neighbours = [grid[i, (j-1)%N], grid[i, (j+1)%N],
                   grid[(i+1)%N, j], grid[(i+1)%N, (j-1)%N],
                   grid[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [grid[i, (j+1)%N], grid[(i+1)%N, j],
                   grid[(i+1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [grid[i, (j-1)%N], grid[(i+1)%N, j],
                   grid[(i+1)%N, (j-1)%N]]
        elif i == N-1:
            if j >= 1 and j < N-1:
                neighbours = [grid[i, (j-1)%N], grid[i, (j+1)%N],
                    grid[(i-1)%N, j], grid[(i-1)%N, (j-1)%N],
                    grid[(i-1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [grid[i, (j+1)%N], grid[(i-1)%N, j],
                    grid[(i-1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [grid[i, (j-1)%N], grid[(i-1)%N, j],
                    grid[(i-1)%N, (j-1)%N]]

    if scoring:
        if i >= 1 and i < N-1:
            if j >= 1 and j < N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[i, (j+1)%N],
                    scoreGrid[(i-1)%N, j], scoreGrid[(i+1)%N, j],
                    scoreGrid[(i-1)%N, (j-1)%N], scoreGrid[(i-1)%N, (j+1)%N],
                    scoreGrid[(i+1)%N, (j-1)%N], scoreGrid[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [scoreGrid[i, (j+1)%N], scoreGrid[(i-1)%N, j],
                    scoreGrid[(i+1)%N, j], scoreGrid[(i-1)%N, (j+1)%N],
                    scoreGrid[(i+1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[(i-1)%N, j],
                    scoreGrid[(i+1)%N, j], scoreGrid[(i-1)%N, (j-1)%N],
                    scoreGrid[(i+1)%N, (j-1)%N]]
        elif i == 0:
            if j >= 1 and j < N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[i, (j+1)%N],
                   scoreGrid[(i+1)%N, j], scoreGrid[(i+1)%N, (j-1)%N],
                   scoreGrid[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [scoreGrid[i, (j+1)%N], scoreGrid[(i+1)%N, j],
                   scoreGrid[(i+1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[(i+1)%N, j],
                   scoreGrid[(i+1)%N, (j-1)%N]]
        elif i == N-1:
            if j >= 1 and j < N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[i, (j+1)%N],
                    scoreGrid[(i-1)%N, j], scoreGrid[(i-1)%N, (j-1)%N],
                    scoreGrid[(i-1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [scoreGrid[i, (j+1)%N], scoreGrid[(i-1)%N, j],
                    scoreGrid[(i-1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[(i-1)%N, j],
                    scoreGrid[(i-1)%N, (j-1)%N]]
    return neighbours

def interconnected_grids(
    N, i, j, grid, grid2, scoreGrid, scoreGrid2, scoring=False):
    if not scoring:
        if i >= 1 and i < N-1:
            if j >= 1 and j < N-1:
                neighbours = [grid[i, (j-1)%N], grid[i, (j+1)%N],
                    grid[(i-1)%N, j], grid[(i+1)%N, j],
                    grid[(i-1)%N, (j-1)%N], grid[(i-1)%N, (j+1)%N],
                    grid[(i+1)%N, (j-1)%N], grid[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [grid[i, (j+1)%N], grid[(i-1)%N, j],
                    grid[(i+1)%N, j], grid[(i-1)%N, (j+1)%N],
                    grid[(i+1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [grid[i, (j-1)%N], grid[(i-1)%N, j],
                    grid[(i+1)%N, j], grid[(i-1)%N, (j-1)%N],
                    grid[(i+1)%N, (j-1)%N]]
        elif i == 0:
            if j >= 1 and j < N-1:
                neighbours = [grid[i, (j-1)%N], grid[i, (j+1)%N],
                   grid[(i+1)%N, j], grid[(i+1)%N, (j-1)%N],
                   grid[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [grid[i, (j+1)%N], grid[(i+1)%N, j],
                   grid[(i+1)%N, (j+1)%N], grid2[0, 0]]
            elif j == N-1:
                neighbours = [grid[i, (j-1)%N], grid[(i+1)%N, j],
                   grid[(i+1)%N, (j-1)%N]]
        elif i == N-1:
            if j >= 1 and j < N-1:
                neighbours = [grid[i, (j-1)%N], grid[i, (j+1)%N],
                    grid[(i-1)%N, j], grid[(i-1)%N, (j-1)%N],
                    grid[(i-1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [grid[i, (j+1)%N], grid[(i-1)%N, j],
                    grid[(i-1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [grid[i, (j-1)%N], grid[(i-1)%N, j],
                    grid[(i-1)%N, (j-1)%N]]

    if scoring:
        if i >= 1 and i < N-1:
            if j >= 1 and j < N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[i, (j+1)%N],
                    scoreGrid[(i-1)%N, j], scoreGrid[(i+1)%N, j],
                    scoreGrid[(i-1)%N, (j-1)%N], scoreGrid[(i-1)%N, (j+1)%N],
                    scoreGrid[(i+1)%N, (j-1)%N], scoreGrid[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [scoreGrid[i, (j+1)%N], scoreGrid[(i-1)%N, j],
                    scoreGrid[(i+1)%N, j], scoreGrid[(i-1)%N, (j+1)%N],
                    scoreGrid[(i+1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[(i-1)%N, j],
                    scoreGrid[(i+1)%N, j], scoreGrid[(i-1)%N, (j-1)%N],
                    scoreGrid[(i+1)%N, (j-1)%N]]
        elif i == 0:
            if j >= 1 and j < N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[i, (j+1)%N],
                   scoreGrid[(i+1)%N, j], scoreGrid[(i+1)%N, (j-1)%N],
                   scoreGrid[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [scoreGrid[i, (j+1)%N], scoreGrid[(i+1)%N, j],
                   scoreGrid[(i+1)%N, (j+1)%N], scoreGrid2[0, 0]]
            elif j == N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[(i+1)%N, j],
                   scoreGrid[(i+1)%N, (j-1)%N]]
        elif i == N-1:
            if j >= 1 and j < N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[i, (j+1)%N],
                    scoreGrid[(i-1)%N, j], scoreGrid[(i-1)%N, (j-1)%N],
                    scoreGrid[(i-1)%N, (j+1)%N]]
            elif j == 0:
                neighbours = [scoreGrid[i, (j+1)%N], scoreGrid[(i-1)%N, j],
                    scoreGrid[(i-1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[(i-1)%N, j],
                    scoreGrid[(i-1)%N, (j-1)%N]]

    # now neighbours for grid2
    if not scoring:
        if i >= 1 and i < N-1:
            if j >= 1 and j < N-1:
                neighbours2 = [grid2[i, (j-1)%N], grid2[i, (j+1)%N],
                    grid2[(i-1)%N, j], grid2[(i+1)%N, j],
                    grid2[(i-1)%N, (j-1)%N], grid2[(i-1)%N, (j+1)%N],
                    grid2[(i+1)%N, (j-1)%N], grid2[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours2 = [grid2[i, (j+1)%N], grid2[(i-1)%N, j],
                    grid2[(i+1)%N, j], grid2[(i-1)%N, (j+1)%N],
                    grid2[(i+1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours2 = [grid2[i, (j-1)%N], grid2[(i-1)%N, j],
                    grid2[(i+1)%N, j], grid2[(i-1)%N, (j-1)%N],
                    grid2[(i+1)%N, (j-1)%N]]
        elif i == 0:
            if j >= 1 and j < N-1:
                neighbours2 = [grid2[i, (j-1)%N], grid2[i, (j+1)%N],
                   grid2[(i+1)%N, j], grid2[(i+1)%N, (j-1)%N],
                   grid2[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours2 = [grid2[i, (j+1)%N], grid2[(i+1)%N, j],
                   grid2[(i+1)%N, (j+1)%N], grid[0, 0]]
            elif j == N-1:
                neighbours2 = [grid2[i, (j-1)%N], grid2[(i+1)%N, j],
                   grid2[(i+1)%N, (j-1)%N]]
        elif i == N-1:
            if j >= 1 and j < N-1:
                neighbours2 = [grid2[i, (j-1)%N], grid2[i, (j+1)%N],
                    grid2[(i-1)%N, j], grid2[(i-1)%N, (j-1)%N],
                    grid2[(i-1)%N, (j+1)%N]]
            elif j == 0:
                neighbours2 = [grid2[i, (j+1)%N], grid2[(i-1)%N, j],
                    grid2[(i-1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours2 = [grid2[i, (j-1)%N], grid2[(i-1)%N, j],
                    grid2[(i-1)%N, (j-1)%N]]

    if scoring:
        if i >= 1 and i < N-1:
            if j >= 1 and j < N-1:
                neighbours2 = [scoreGrid2[i, (j-1)%N], scoreGrid2[i, (j+1)%N],
                    scoreGrid2[(i-1)%N, j], scoreGrid2[(i+1)%N, j],
                    scoreGrid2[(i-1)%N, (j-1)%N], scoreGrid2[(i-1)%N, (j+1)%N],
                    scoreGrid2[(i+1)%N, (j-1)%N], scoreGrid2[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours2 = [scoreGrid2[i, (j+1)%N], scoreGrid2[(i-1)%N, j],
                    scoreGrid2[(i+1)%N, j], scoreGrid2[(i-1)%N, (j+1)%N],
                    scoreGrid2[(i+1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours2 = [scoreGrid2[i, (j-1)%N], scoreGrid2[(i-1)%N, j],
                    scoreGrid2[(i+1)%N, j], scoreGrid2[(i-1)%N, (j-1)%N],
                    scoreGrid2[(i+1)%N, (j-1)%N]]
        elif i == 0:
            if j >= 1 and j < N-1:
                neighbours2 = [scoreGrid2[i, (j-1)%N], scoreGrid2[i, (j+1)%N],
                   scoreGrid2[(i+1)%N, j], scoreGrid2[(i+1)%N, (j-1)%N],
                   scoreGrid2[(i+1)%N, (j+1)%N]]
            elif j == 0:
                neighbours2 = [scoreGrid2[i, (j+1)%N], scoreGrid2[(i+1)%N, j],
                   scoreGrid2[(i+1)%N, (j+1)%N], scoreGrid[0, 0]]
            elif j == N-1:
                neighbours2 = [scoreGrid2[i, (j-1)%N], scoreGrid2[(i+1)%N, j],
                   scoreGrid2[(i+1)%N, (j-1)%N]]
        elif i == N-1:
            if j >= 1 and j < N-1:
                neighbours2 = [scoreGrid2[i, (j-1)%N], scoreGrid2[i, (j+1)%N],
                    scoreGrid2[(i-1)%N, j], scoreGrid2[(i-1)%N, (j-1)%N],
                    scoreGrid2[(i-1)%N, (j+1)%N]]
            elif j == 0:
                neighbours2 = [scoreGrid2[i, (j+1)%N], scoreGrid2[(i-1)%N, j],
                    scoreGrid2[(i-1)%N, (j+1)%N]]
            elif j == N-1:
                neighbours2 = [scoreGrid2[i, (j-1)%N], scoreGrid2[(i-1)%N, j],
                    scoreGrid2[(i-1)%N, (j-1)%N]]
    return neighbours, neighbours2


def freq_calc(grid, grid2, N, b):
    '''This is designed to calculate the frequency of cooperators
       at each timestep, then plot how this changes over time'''

    time_list = np.arange(101)
    coop_freq_list = []
    # present a progress bar in terminal for confidence in long running simulations
    bar = IncrementalBar('Running', max=len(time_list),
                        suffix='%(percent).1f%% complete - Time elapsed: %(elapsed)ds - Estimated time remaining: %(eta)ds')
    for timestep, time in enumerate(time_list):
        # copy grid since we require 8 neighbours for calculation
        newGrid = grid.copy()
        newGrid2 = grid2.copy()
        #scoreGrid = scoreGrid.copy()
        scoreGrid = np.zeros(N*N).reshape(N, N)
        scoreGrid2 = np.zeros(N*N).reshape(N, N)
        # a cooperator in the grid is denoted by a 1, defector by a 0
        # so we count the amount of 1s in the grid at each step
        # then convert this to a frequency and add it to the list
        # count freq in grid 2 to test if connections work
        player, coop_count = np.unique(grid2, return_counts=True)
        if len(coop_count) == 1: # if count is zero the [1] index will not exist
            coop_count = np.append(coop_count, 0) # this if handles the error
        coop_freq = coop_count[1] / (N*N) # index 1 accesses cooperators
        coop_freq_list.append(coop_freq)

        # scoring loop
        # each player plays their neighbour in turn
        # and we go line by line
        for i in range(N):
            for j in range(N):
                # reset the current players score
                score = 0
                score2 = 0
                # initiate a list of all their neighbours

                neighbours, neighbours2 = interconnected_grids(N, i, j, grid, grid2,
                                         scoreGrid, scoreGrid2, scoring=False)
                # apply game rules
                # coop playing themself will always get payoff 1
                # no other self-interaction gives payoff
                # remove this line to remove self-interaction

                if grid[i, j] == 1:
                    score += 1

                # play against each neighbour
                for index, elem in enumerate(neighbours):
                    if elem == grid[i, j] and elem == COOP:
                        score += 1
                    elif elem == grid[i, j] and elem == DEFEC:
                        score += 0
                    elif grid[i,j] == COOP and elem == DEFEC:
                        score += 0
                    elif grid[i,j] == DEFEC and elem == COOP:
                        score += b
                    # after playing each neighbour,
                    # update the score on the grid
                    if index == len(neighbours)-1:
                        scoreGrid[i, j] = score

                # same for other group
                if grid2[i, j] == 1:
                    score2 += 1
                # play against each neighbour
                for index, elem in enumerate(neighbours2):
                    if elem == grid2[i, j] and elem == COOP:
                        score2 += 1
                    elif elem == grid2[i, j] and elem == DEFEC:
                        score2 += 0
                    elif grid2[i,j] == COOP and elem == DEFEC:
                        score2 += 0
                    elif grid2[i,j] == DEFEC and elem == COOP:
                        score2 += b
                    # after playing each neighbour,
                    # update the score on the grid
                    if index == len(neighbours2)-1:
                        scoreGrid2[i, j] = score2

        # now each location has a score
        # we compare each score to its neighbours
        # and go line by line again
        for i in range(N):
            for j in range(N):
                # create a list of the neighbours scores to compare with

                scoreneighbours, scoreneighbours2 = interconnected_grids(N, i, j, grid, grid2,
                                         scoreGrid, scoreGrid2, scoring=True)
                # get location of highest score obtained by neighbours
                sort_scoreneighbours = sorted(scoreneighbours)
                hscore_index = scoreneighbours.index(sort_scoreneighbours[-1])

                # compare current players score with highest neighbours score
                # if current players score is higher, they stay as the player
                # in the next round
                if scoreGrid[i, j] >= sort_scoreNeighbours[-1]:
                    newGrid[i, j] = grid[i, j]

                # if a neighbour has a higher score, we find out which player
                # and that player takes over this cell in the next round
                elif scoreGrid[i, j] < sort_scoreNeighbours[-1]:
                    select, select2 = interconnected_grids(N, i, j, grid, grid2,
                                             scoreGrid, scoreGrid2)
                    newGrid[i, j] = select[hscore_index]


                # now do the same for grid2
                sort_scoreneighbours2 = sorted(scoreneighbours2)
                hscore_index2 = scoreneighbours2.index(sort_scoreneighbours2[-1])

                # compare current players score with highest neighbours score
                # if current players score is higher, they stay as the player
                # in the next round
                if scoreGrid2[i, j] >= sort_scoreNeighbours2[-1]:
                    newGrid2[i, j] = grid2[i, j]

                # if a neighbour has a higher score, we find out which player
                # and that player takes over this cell in the next round
                elif scoreGrid2[i, j] < sort_scoreNeighbours2[-1]:
                    select, select2 = interconnected_grids(N, i, j, grid, grid2,
                                             scoreGrid, scoreGrid2)
                    newGrid2[i, j] = select2[hscore_index2]


        grid[:] = newGrid[:]
        grid2[:] = newGrid2[:] # replace current grids with new grid
        bar.next() # update progress bar
    bar.finish()
    # now sort out the plotting, make it pretty
    plt.style.use('ggplot')
    plt.figure(figsize=(9,5))
    freqPlot = plt.plot(time_list,coop_freq_list, linewidth=0.7)
    plt.title('b = ' + str(b))
    plt.axhline(y=0.31776617, linestyle='--', color='gray', linewidth=0.7)
    plt.xlabel('Time')
    plt.ylabel('Frequency of Cooperators, x')
    plt.xlim(-1, len(time_list))
    #plt.ylim(0, 0.7)
    plt.show()


def update(frameNum, img, grid, N, b):
    # copy grid since we require 8 neighbors for calculation
    newGrid = grid.copy()
    #scoreGrid = scoreGrid.copy()
    scoreGrid = np.zeros(N*N).reshape(N, N)

    # scoring loop
    # each player plays their neighbour in turn
    # and we go line by line
    for i in range(N):
        for j in range(N):
            # reset the current players score
            score = 0
            # initiate a list of all their neighbours
            # this is toroidal kings move condtions
            neighbours = kings_neighbours(N, i, j, grid, scoreGrid)
            #neighbours = fixed_bc_kings_neighbours(N, i, j, grid, scoreGrid)
            # apply game rules
            # coop playing themself will always get payoff 1
            # no other self-interaction gives payoff
            if grid[i, j] == 1:
                score += 1
            # play against each neighbour
            for index, elem in enumerate(neighbours):
                if elem == grid[i, j] and elem == COOP:
                    score += 1
                elif elem == grid[i, j] and elem == DEFEC:
                    score += 0
                elif grid[i,j] == COOP and elem == DEFEC:
                    score += 0
                elif grid[i,j] == DEFEC and elem == COOP:
                    score += b
                # after playing each neighbour,
                # update the score on the grid
                if index == 7:
                    scoreGrid[i, j] = score

    # now each location has a score
    # we compare each score to its neighbours
    # and go line by line again
    for i in range(N):
        for j in range(N):
            # create a list of the neighbours scores to compare with
            scoreneighbours = kings_neighbours(N, i, j, grid, scoreGrid, scoring=True)
            #scoreneighbours = fixed_bc_kings_neighbours(N, i, j, grid, scoreGrid, scoring=True)
            # get location of highest score obtained by neighbours
            sort_scoreneighbours = sorted(scoreneighbours)
            hscore_index = scoreneighbours.index(sort_scoreneighbours[-1])

                # compare current players score with highest neighbours score
                # if current players score is higher, they stay as the player
                # in the next round
                if scoreGrid[i, j] >= sort_scoreNeighbours[-1]:
                    newGrid[i, j] = grid[i, j]

                # if a neighbour has a higher score, we find out which player
                # and that player takes over this cell in the next round
                elif scoreGrid[i, j] < sort_scoreNeighbours[-1]:
                    select = fixed_bc_kings_neighbours(N, i, j, grid, scoreGrid)
                    newGrid[i, j] = select[hscore_index]

    # finally update data
    # in format for visualisation
    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return img,


def main():
    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored
    # parse arguments
    parser = argparse.ArgumentParser(
    description='''This will produce new, untested simulations of
    evolutionary games on complex networks''',
    formatter_class=CustomHelpFormatter)
    # add arguments
    parser.add_argument('--grid-size', '-N', dest='N', required=True,
    help='Size of grid to run the simulation on. This will generate a N x N square grid.')
    parser.add_argument('--movfile', dest='movfile', required=False,
                        help='''Saves the animation to filename/path you provide.
                        Saves as <arg>_<b arg>.gif or
                        as <arg>_<b arg>.png if used with -f''')
    parser.add_argument('--interval', '-i', dest='interval', required=False,
                        help='''Interval between animation updates in milliseconds.
                        200 is default if this switch is not specified''')
    parser.add_argument('--glider', '-g', dest='glider',
                        action='store_true', required=False,
                        help='Places a glider strucure in an otherwise empty grid')
    parser.add_argument('--rotator', '-r', dest='rotator',
                        action='store_true', required=False,
                        help='Places a rotator strucure in an otherwise empty grid')
    parser.add_argument('--grower', '-G', dest='grower',
                        action='store_true', required=False,
                        help='Places a grower strucure in an otherwise empty grid')
    parser.add_argument('-b', dest='b', required=True,
                        help='This is the advantage for defectors.')
    parser.add_argument('--frequency', '-f', dest='frequency',
                        action='store_true', required=False,
                        help='Produce frequency of cooperators graph')
    args = parser.parse_args()

    #intitate inputs
    b = float(args.b)
    N = int(args.N)

    # set animation update interval
    updateInterval = 200
    if args.interval:
        updateInterval = int(args.interval)

    # intitate grid and check for specific setups declared e.g. glider
    grid = np.array([])
    if args.glider:
        grid = np.zeros(N*N).reshape(N, N)
        addGlider(15, 5, grid)
    elif args.rotator:
        grid = np.zeros(N*N).reshape(N, N)
        addRotator(10, 10, grid)
    elif args.grower:
        grid = np.zeros(N*N).reshape(N, N)
        addGrower(8, 8, grid)
    else:
        grid = randomGrid(N)
        grid2= np.ones(N*N).reshape(N, N) # start blank to show invasion
        grid2[10, 10] = 0

    # this will make the color scheme of the visual plots
    # identical to that used in the paper
    # just add cmap=cm arg to imshow
    colors = [(1, 0, 0), (0, 0, 1)]
    cmap_name = 'nowak_scheme'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)

    if args.frequency:
        freq_calc(grid, grid2, N, b)
        if args.movfile:
            plt.savefig(args.movfile+'_b='+str(b)+'.png')
    else:
        # set up animation
        fig, ax = plt.subplots()
        img = ax.imshow(grid, interpolation='nearest', cmap='binary')
        plt.colorbar(img, ax=ax)
        plt.title('Black denotes Cooperator')
        # display animation length in seconds is frames * interval /1000
        ani = animation.FuncAnimation(fig, update,
                                      fargs=(img, grid, N, b, ),
                                      frames = 20,
                                      interval=updateInterval,
                                      save_count=50)

        # saved animation duration in seconds is frames * (1 / fps)
        # set output file
        writergif = animation.PillowWriter(fps=5)
        if args.movfile:
            ani.save(args.movfile+'.gif',writer=writergif)

        plt.show()


# call main
if __name__ == '__main__':
    main()
