"""
Author: Adam W
This will produce simulations of 'vaccinated' cells integration
into a generally unvaccinated population. They are at random, 
fixed locations. These will be run on a complex network which 
will be 2 interconnected populations with fixed boundaries.
"""

import sys, argparse, time, threading
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

VACC = 2
COOP = 1
DEFEC = 0
vals = [COOP, DEFEC, VACC]

def randomGrid(N, cf=0.6, df=0.1, vf=0.3):
    """returns a grid of NxN random values."""
    return np.random.choice(vals, N*N, p=[cf, df, vf]).reshape(N, N)

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
                if i == int(N/2) and j == int(N/2):
                    neighbours = [grid[i, (j-1)%N], grid[i, (j+1)%N],
                        grid[(i-1)%N, j], grid[(i+1)%N, j],
                        grid[(i-1)%N, (j-1)%N], grid[(i-1)%N, (j+1)%N],
                        grid[(i+1)%N, (j-1)%N], grid[(i+1)%N, (j+1)%N],
                        grid2[int(N/2), int(N/2)]]
                else:
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
                if i == int(N/2) and j == int(N/2):
                    neighbours = [scoreGrid[i, (j-1)%N], scoreGrid[i, (j+1)%N],
                        scoreGrid[(i-1)%N, j], scoreGrid[(i+1)%N, j],
                        scoreGrid[(i-1)%N, (j-1)%N], scoreGrid[(i-1)%N, (j+1)%N],
                        scoreGrid[(i+1)%N, (j-1)%N], scoreGrid[(i+1)%N, (j+1)%N],
                        scoreGrid2[int(N/2), int(N/2)]]
                else:
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

    # now neighbours for grid2
    if not scoring:
        if i >= 1 and i < N-1:
            if j >= 1 and j < N-1:
                if i == int(N/2) and j == int(N/2):
                    neighbours2 = [grid2[i, (j-1)%N], grid2[i, (j+1)%N],
                        grid2[(i-1)%N, j], grid2[(i+1)%N, j],
                        grid2[(i-1)%N, (j-1)%N], grid2[(i-1)%N, (j+1)%N],
                        grid2[(i+1)%N, (j-1)%N], grid2[(i+1)%N, (j+1)%N],
                        grid[int(N/2), int(N/2)]]
                else:
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
                   grid2[(i+1)%N, (j+1)%N]]
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
                if i == int(N/2) and j == int(N/2):
                    neighbours2 = [scoreGrid2[i, (j-1)%N], scoreGrid2[i, (j+1)%N],
                        scoreGrid2[(i-1)%N, j], scoreGrid2[(i+1)%N, j],
                        scoreGrid2[(i-1)%N, (j-1)%N], scoreGrid2[(i-1)%N, (j+1)%N],
                        scoreGrid2[(i+1)%N, (j-1)%N], scoreGrid2[(i+1)%N, (j+1)%N],
                        scoreGrid[int(N/2), int(N/2)]]
                else:
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
                   scoreGrid2[(i+1)%N, (j+1)%N]]
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


def freq_calc(grid, grid2, N, b, c, d, v, generations):
    '''This is designed to calculate the frequency of cooperators
       at each timestep, then plot how this changes over time'''

    time_list = np.arange(generations+1)
    defec_freq_list = []
    defec_freq_list2 = []
    coop_freq_list = []
    coop_freq_list2 = []
    vacc_freq_list = []
    vacc_freq_list2 = []

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
        player, count = np.unique(grid, return_counts=True)

        # this set of condtionals handles cases when counts go to zero
        # when the counts goes to zero it is not included in the list
        # so this set will insert those values into the list
        if len(count) == 2:
            if 0 not in player:
                count = np.insert(count, 0, 0)
            elif 1 not in player:
                count = np.insert(count, 1, 0)
            elif 2 not in player:
                count = np.insert(count, 2, 0)
        if len(count) == 1:
            if player[0] == 0:
                count = np.insert(count, 1, 0)
                count = np.append(count, 2)
            if player[0] == 1:
                count = np.insert(count, 0, 0)
                count = np.append(count, 2)
            if player[0] == 2:
                count = np.insert(count, 0, 0)
                count = np.insert(count, 1, 0)

        defec_freq = count[0] / (N*N) # index 1 is always defectors
        defec_freq_list.append(defec_freq)
        coop_freq = count[1] / (N*N) # index 1 is always cooperators
        coop_freq_list.append(coop_freq)
        vacc_freq = count[2] / (N*N) # index 2 is always vaccinated
        vacc_freq_list.append(vacc_freq)

        # count freq in grid 2
        player2, count2 = np.unique(grid2, return_counts=True)
        # this set of condtionals handles cases when counts go to zero
        # when the counts goes to zero it is not included in the list
        # so this set will insert those values into the list
        if len(count2) == 2:
            if 0 not in player2:
                count2 = np.insert(count2, 0, 0)
            elif 1 not in player2:
                count2 = np.insert(count2, 1, 0)
            elif 2 not in player2:
                count2 = np.insert(count2, 2, 0)
        if len(count2) == 1:
            if player2[0] == 0:
                count2 = np.insert(count2, 1, 0)
                count2 = np.append(count2, 2)
            if player2[0] == 1:
                count2 = np.insert(count2, 0, 0)
                count2 = np.append(count2, 2)
            if player2[0] == 2:
                count2 = np.insert(count2, 0, 0)
                count2 = np.insert(count2, 1, 0)

        defec_freq2 = count2[0] / (N*N) # index 1 is always defectors
        defec_freq_list2.append(defec_freq2)
        coop_freq2 = count2[1] / (N*N) # index 1 is always cooperators
        coop_freq_list2.append(coop_freq2)
        vacc_freq2 = count2[2] / (N*N) # index 2 is always vaccinated
        vacc_freq_list2.append(vacc_freq2)

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
                    elif elem == grid[i, j] and elem == VACC:
                        score += v
                    elif grid[i, j] == COOP and elem == VACC:
                        score += c
                    elif grid[i, j] == DEFEC and elem == VACC:
                        score += d
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
                    elif elem == grid2[i, j] and elem == VACC:
                        score2 += v
                    elif grid2[i, j] == COOP and elem == VACC:
                        score2 += c
                    elif grid2[i, j] == DEFEC and elem == VACC:
                        score2 += d
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
                scoreNeighbours, scoreNeighbours2 = interconnected_grids(N, i, j, grid, grid2,
                                         scoreGrid, scoreGrid2, scoring=True)
                # generate list of neighbours player types
                select, select2 = interconnected_grids(N, i, j, grid, grid2,
                                             scoreGrid, scoreGrid2)

                # this loop stops vaccinated people from spreading
                # by ensuring all vaccinated neighbours have lowest possible score
                for index, elem in enumerate(select):
                    if elem == 2:
                        scoreNeighbours[index] = 0

                # get location of highest score obtained by neighbours
                sort_scoreNeighbours = sorted(scoreNeighbours)
                hscore_index = scoreNeighbours.index(sort_scoreNeighbours[-1])

                # compare current players score with highest neighbours score
                # if current players score is higher, they stay as the player
                # in the next round
                if scoreGrid[i, j] >= sort_scoreNeighbours[-1]:
                    newGrid[i, j] = grid[i, j]

                # if a neighbour has a higher score, we find out which player
                # and that player takes over this cell in the next round
                elif scoreGrid[i, j] < sort_scoreNeighbours[-1]:
                    newGrid[i, j] = select[hscore_index]

                # this makes sure vaccinated cells do not disappear
                if grid[i, j] == 2:
                    newGrid[i, j] = grid[i, j]

                ''' now do the same for grid2 '''

                # this loop stops vaccinated people from spreading
                # by ensuring all vaccinated neighbours have lowest possible score
                for index, elem in enumerate(select2):
                    if elem == 2:
                        scoreNeighbours2[index] = 0

                sort_scoreNeighbours2 = sorted(scoreNeighbours2)
                hscore_index2 = scoreNeighbours2.index(sort_scoreNeighbours2[-1])

                # compare current players score with highest neighbours score
                # if current players score is higher, they stay as the player
                # in the next round
                if scoreGrid2[i, j] >= sort_scoreNeighbours2[-1]:
                    newGrid2[i, j] = grid2[i, j]

                # if a neighbour has a higher score
                # that player takes over this cell in the next round
                elif scoreGrid2[i, j] < sort_scoreNeighbours2[-1]:
                    newGrid2[i, j] = select2[hscore_index2]

                # this makes sure vaccinated cells do not disappear
                if grid2[i, j] == 2:
                    newGrid2[i, j] = grid2[i, j]


        grid[:] = newGrid[:]
        grid2[:] = newGrid2[:] # replace current grids with new grid
        bar.next() # update progress bar
    bar.finish()
    # now sort out the plotting
    plt.style.use('ggplot') # pretty format
    plt.figure(figsize=(9,5))
    plt.plot(time_list, coop_freq_list, linewidth=0.7, label='Cooperator Frequency')
    plt.plot(time_list, defec_freq_list, linewidth=0.7, label='Defector Frequency')
    plt.plot(time_list, vacc_freq_list, linewidth=0.7, label='Vaccinated Frequency')
    plt.title('b = ' + str(b) + '   N = ' + str(N) + '  Grid 1')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Frequency of player')
    plt.xlim(-1, len(time_list))


    plt.figure(figsize=(9,5))
    plt.plot(time_list, coop_freq_list2, linewidth=0.7, label='Cooperator Frequency')
    plt.plot(time_list, defec_freq_list2, linewidth=0.7, label='Defector Frequency')
    plt.plot(time_list, vacc_freq_list2, linewidth=0.7, label='Vaccinated Frequency')
    plt.title('b = ' + str(b) + '   N = ' + str(N) + '  Grid 2')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Frequency of Cooperators, x')
    plt.xlim(-1, len(time_list))

    plt.show()


def update(frameNum, img, grid, grid2, N, b, c, d, v, gridNo):
    # copy grid since we require 8 neighbors for calculation
    newGrid = grid.copy()
    newGrid2 = grid2.copy()
    scoreGrid = np.zeros(N*N).reshape(N, N)
    scoreGrid2 = np.zeros(N*N).reshape(N, N)

    # update data
    # in format for visualisation
    # do this first to avoid frame loss
    if gridNo == 1:
        img.set_data(newGrid)
    elif gridNo == 2:
        img.set_data(newGrid2)

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
                elif elem == grid[i, j] and elem == VACC:
                    score += v
                elif grid[i, j] == COOP and elem == VACC:
                    score += c
                elif grid[i, j] == DEFEC and elem == VACC:
                    score += d
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
                elif elem == grid2[i, j] and elem == VACC:
                    score2 += v
                elif grid2[i, j] == COOP and elem == VACC:
                    score2 += c
                elif grid2[i, j] == DEFEC and elem == VACC:
                    score2 += d
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
            scoreNeighbours, scoreNeighbours2 = interconnected_grids(N, i, j, grid, grid2,
                                     scoreGrid, scoreGrid2, scoring=True)
            # generate list of neighbours player types
            select, select2 = interconnected_grids(N, i, j, grid, grid2,
                                         scoreGrid, scoreGrid2)

            # this loop stops vaccinated people from spreading
            # by ensuring all vaccinated neighbours have lowest possible score
            for index, elem in enumerate(select):
                if elem == 2:
                    scoreNeighbours[index] = 0

            # get location of highest score obtained by neighbours
            sort_scoreNeighbours = sorted(scoreNeighbours)
            hscore_index = scoreNeighbours.index(sort_scoreNeighbours[-1])

            # compare current players score with highest neighbours score
            # if current players score is higher, they stay as the player
            # in the next round
            if scoreGrid[i, j] >= sort_scoreNeighbours[-1]:
                newGrid[i, j] = grid[i, j]

            # if a neighbour has a higher score, we find out which player
            # and that player takes over this cell in the next round
            elif scoreGrid[i, j] < sort_scoreNeighbours[-1]:
                newGrid[i, j] = select[hscore_index]

            # this makes sure vaccinated cells do not disappear
            if grid[i, j] == 2:
                newGrid[i, j] = grid[i, j]

            ''' now do the same for grid2 '''

            # this loop stops vaccinated people from spreading
            # by ensuring all vaccinated neighbours have lowest possible score
            for index, elem in enumerate(select2):
                if elem == 2:
                    scoreNeighbours2[index] = 0

            sort_scoreNeighbours2 = sorted(scoreNeighbours2)
            hscore_index2 = scoreNeighbours2.index(sort_scoreNeighbours2[-1])

            # compare current players score with highest neighbours score
            # if current players score is higher, they stay as the player
            # in the next round
            if scoreGrid2[i, j] >= sort_scoreNeighbours2[-1]:
                newGrid2[i, j] = grid2[i, j]

            # if a neighbour has a higher score
            # and that player takes over this cell in the next round
            elif scoreGrid2[i, j] < sort_scoreNeighbours2[-1]:
                newGrid2[i, j] = select2[hscore_index2]

            # this makes sure vaccinated cells do not disappear
            if grid2[i, j] == 2:
                newGrid2[i, j] = grid2[i, j]

    # temporary grid is now constructed so becomes the new grid
    grid[:] = newGrid[:]
    grid2[:] = newGrid2[:] # replace current grids with new grid
    return img,

def visualisation(grid, grid2, N, b, c, d, v, gridNo, updateInterval, movfile):
    # this will make the color scheme of the visual plots
    # identical to that used in Nowak & May paper
    # just add cmap=cm arg to imshow
    colors = [(1, 0, 0), (0, 0, 1)]
    cmap_name = 'nowak_scheme'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)
    if gridNo == 1 or gridNo is None:
        gridNo = 1
        fig, ax = plt.subplots()
        img = ax.imshow(grid, vmin=0.0, vmax=2.0, interpolation='none', cmap='binary')
        plt.colorbar(img, ax=ax)
        plt.title('''Black: Vaccinated, Grey: Cooperator, White: Defector
                  b = ''' + str(b) + ' Grid' + str(gridNo))
        # display animation length in seconds is frames * interval /1000
        ani = animation.FuncAnimation(fig, update,
                                      fargs=(img, grid, grid2, N, b, c, d, v, gridNo, ),
                                      frames = 40,
                                      interval=updateInterval,
                                      save_count=50)

        # saved animation duration in seconds is frames * (1 / fps)
        # set output file
        writergif = animation.PillowWriter(fps=4)
        if movfile:
            ani.save(movfile+'('+str(gridNo)+').gif',writer=writergif)
        plt.show()
    # set up animation for grid 2
    elif gridNo == 2:
        fig, ax = plt.subplots()
        img = ax.imshow(grid2, vmin=0.0, vmax=2.0, interpolation='none', cmap='binary')
        plt.colorbar(img, ax=ax)
        plt.title('''Black: Vaccinated, Grey: Cooperator, White: Defector
                  b = ''' + str(b) + ' Grid' + str(gridNo))
        # display animation length in seconds is frames * interval /1000
        ani = animation.FuncAnimation(fig, update,
                                      fargs=(img, grid, grid2, N, b, c, d, v, gridNo, ),
                                      frames = 40,
                                      interval=updateInterval,
                                      save_count=50)

        writergif = animation.PillowWriter(fps=4)
        if movfile:
            ani.save(movfile+'('+str(gridNo)+').gif',writer=writergif)
        plt.show()
    else:
        raise Exception('You must specify which grid to visualise with -gn flag.')

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
    parser.add_argument('--lone-defector', '-ld', dest='lone_d',
                        action='store_true', required=False,
                        help='Introduce a lone defector in the middle of the grid')
    parser.add_argument('--grid-vis', '-gn', dest='gridNo', type=int, required=False,
                        help='''Choose which grid to visualise, 1 or 2.
                        Default is grid 1''')
    args = parser.parse_args()

    #intitate inputs
    b = float(args.b)
    N = int(args.N)
    movfile = args.movfile
    gridNo = args.gridNo

    v = 2
    c = 1
    d = 1

    cf = 0.6
    df = 0.3
    vf = 0.1

    # set animation update interval
    updateInterval = generations = 200
    if args.interval:
        updateInterval = generations = int(args.interval)

    # intitate grid and check for specific setups declared e.g. glider
    grid = np.array([])
    if args.glider:
        grid = np.zeros(N*N).reshape(N, N)
        addGlider(17, 8, grid)
        grid2 = np.zeros(N*N).reshape(N, N)
    elif args.rotator:
        grid = np.zeros(N*N).reshape(N, N)
        addRotator(10, 10, grid)
    elif args.grower:
        grid = np.zeros(N*N).reshape(N, N)
        addGrower(9, 9, grid)
        grid2 = np.ones(N*N).reshape(N, N)
    elif args.lone_d:
        grid = np.ones(N*N).reshape(N, N)
        grid2 = np.ones(N*N).reshape(N, N) # start blank to show invasion
        grid[int(N/2), int(N/2)] = 0
        grid2[3:7, 3:7] = 2

    else:
        grid = randomGrid(N)
        grid2 = randomGrid(N, cf, df, vf)


    if args.frequency:
        freq_calc(grid, grid2, N, b, c, d, v, generations)
        if args.movfile:
            plt.savefig(args.movfile+'_b='+str(b)+'.png')
    else:
        # set up animation
        visualisation(grid, grid2, N, b, c, d, v, gridNo, updateInterval, movfile)
        #visualisation(grid, grid2, N, b, gridNo, updateInterval, movfile)



# call main
if __name__ == '__main__':
    main()
