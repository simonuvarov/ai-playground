import numpy as np


def random_argmax(a):
    '''
    Return the index of the maximum value in the array.
    This function is useful for choosing a random action when
    there are multiple actions with the same maximum value.
    '''
    return np.random.choice(np.where(a == a.max())[0])
