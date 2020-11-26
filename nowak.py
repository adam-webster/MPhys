"""
Modified from conway.py by Mahesh Venkitachalam.
Author: Adam W
This will replicate results from Nowak & May
1993 paper 'The Spatial Dilemmas of Evolution'
"""

import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

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
            # could change this to an external function
            # so then we can choose boundary conditions
            # this is toroidal kings move condtions
            neighbours = [grid[i, (j-1)%N], grid[i, (j+1)%N],
               grid[(i-1)%N, j], grid[(i+1)%N, j],
               grid[(i-1)%N, (j-1)%N], grid[(i-1)%N, (j+1)%N],
               grid[(i+1)%N, (j-1)%N], grid[(i+1)%N, (j+1)%N]]

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
            scoreNeighbours = [scoreGrid[i, (j-1)%N], scoreGrid[i, (j+1)%N],
               scoreGrid[(i-1)%N, j], scoreGrid[(i+1)%N, j],
               scoreGrid[(i-1)%N, (j-1)%N], scoreGrid[(i-1)%N, (j+1)%N],
               scoreGrid[(i+1)%N, (j-1)%N], scoreGrid[(i+1)%N, (j+1)%N]]

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
            elif sort_scoreNeighbours[-1] > scoreGrid[i, j]:
                if hscore_index == 0:
                    newGrid[i, j] = grid[i, (j-1)%N]
                elif hscore_index == 1:
                    newGrid[i, j] = grid[i, (j+1)%N]
                elif hscore_index == 2:
                    newGrid[i, j] = grid[(i-1)%N, j]
                elif hscore_index == 3:
                    newGrid[i, j] = grid[(i+1)%N, j]
                elif hscore_index == 4:
                    newGrid[i, j] = grid[(i-1)%N, (j-1)%N]
                elif hscore_index == 5:
                    newGrid[i, j] = grid[(i-1)%N, (j+1)%N]
                elif hscore_index == 6:
                    newGrid[i, j] = grid[(i+1)%N, (j-1)%N]
                elif hscore_index == 7:
                    newGrid[i, j] = grid[(i+1)%N, (j+1)%N]


    # finally update data
    # in format for visualisation
    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return img,


def main():
    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored
    # i'll add some kind of help function to this
    # parse arguments
    parser = argparse.ArgumentParser(description="Runs Conway's Game of Life simulation.")
  # add arguments
    parser.add_argument('--grid-size', '-N', dest='N', required=True)
    parser.add_argument('--mov-file', dest='movfile', required=False)
    parser.add_argument('--interval', '-i', dest='interval', required=False)
    parser.add_argument('--glider', '-g', dest='glider', action='store_true', required=False)
    parser.add_argument('--rotator', '-r', dest='rotator', action='store_true', required=False)
    parser.add_argument('--grower', '-G', dest='grower', action='store_true', required=False)
    parser.add_argument('-b', dest='b', required=True)
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
        addGlider(15, 1, grid)
    elif args.rotator:
        grid = np.zeros(N*N).reshape(N, N)
        addRotator(10, 10, grid)
    elif args.grower:
        grid = np.zeros(N*N).reshape(N, N)
        addGrower(8, 8, grid)
    else:
        grid = randomGrid(N)

    '''
    # for debugging
    grid = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])
    '''


    # this will make the color scheme of the visual plots
    # identical to that used in the paper
    # just add cmap=cm arg to imshow
    colors = [(1, 0, 0), (0, 0, 1)]
    cmap_name = 'nowak_scheme'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)

    # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest', cmap='binary')
    plt.colorbar(img, ax=ax)
    plt.title('Black denotes Cooperator')
    ani = animation.FuncAnimation(fig, update,
                                  fargs=(img, grid, N, b, ),
                                  frames = 10,
                                  interval=updateInterval,
                                  save_count=50)

    # # of frames?
    # set output file
    if args.movfile:
        ani.save(args.movfile, fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()

# call main
if __name__ == '__main__':
    main()
