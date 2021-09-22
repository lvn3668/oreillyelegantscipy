
# Game of Life solution using Numpy
import numpy as np
from scipy import ndimage as ndi

# A live cell with 0 / 1 neighbor dies
# A live cell with 2/3 live neighbor lives for next generation
# A live cell with 4/more neighbors dies
# A dead cell with exactly 3 neighbors comes alive



def next_generation(MatrixOfGameOfLife):
    Neighbors: int = (MatrixOfGameOfLife[0:-2, 0:-2] + MatrixOfGameOfLife[0: -2, 1:-1] + MatrixOfGameOfLife[0:-2, 2:] + MatrixOfGameOfLife[1:-1, 0:-2] + MatrixOfGameOfLife[1:-1, 2:] + MatrixOfGameOfLife[2:, 0:-2] + MatrixOfGameOfLife[2:, 1:-1] + MatrixOfGameOfLife[2:,

                                                                                                                  2:])
    # A cell comes back alive because it is dead AND has exactly 3 neighbors
    birth = (Neighbors == 3) & (MatrixOfGameOfLife[1:-1, 1:-1] == 0)
    # Cell with 2 or 3 neighbors survives for next generation
    survive = ((Neighbors == 2) | (Neighbors == 3)) & (MatrixOfGameOfLife[1:-1, 1:-1] == 1)
    MatrixOfGameOfLife[...] = 0
    MatrixOfGameOfLife[1:-1, 1:-1][birth | survive] = 1
    return MatrixOfGameOfLife


def nextgen_filter(values):
    values = values.astype(int)
    center = values[len(values) // 2]
    neighbors_count = np.sum(values) - center
    if neighbors_count == 3 or (center and neighbors_count == 2):
        return 1.
    else:
        return 0.


def next_generation(board):
    return ndi.generic_filter(board, nextgen_filter, size=3, mode='constant')

# Toroidal:  Ends wrap around
def next_generation_toroidal(board):
    return ndi.generic_filter(board, nextgen_filter, size=3, mode='wrap')

# Board size is 50 by 50
random_board = np.random.randint(0, 2, size=(50, 50))
print(random_board)
n_generations = 100
for generation in range(n_generations):
    random_board = next_generation_toroidal(random_board)
    print("Generation ", generation)
    print(random_board)
