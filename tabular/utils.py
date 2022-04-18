import numpy as np


def to_grid_coordinates(state):
    '''
    Converts a state to a row, column matrix
    '''
    return state % 4, state // 4


def to_linear_coordinates(row, col):
    '''
    Converts a row, column matrix to a state
    '''
    return row * 4 + col


def print_as_grid(state_values):
    '''
    Prints array values as a grid
    '''
    for row in range(4):
        for col in range(4):
            state = to_linear_coordinates(row, col)
            print("{:.4f}".format(state_values[state]), end=' ')
        print()


def random_argmax(a):
    '''
    Return the index of the maximum value in the array.
    This function is useful for choosing a random action when
    there are multiple actions with the same maximum value.
    '''
    return np.random.choice(np.where(a == a.max())[0])
