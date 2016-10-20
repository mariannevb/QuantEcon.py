"""
Author: Daisuke Oyama

Compute mixed Nash equilibria of a 2-player normal form game by the
Lemke-Howson algorithm.

References
----------
B. von Stengel, "Equilibrium Computation for Two-Player Games in
Strategic and Extensive Form," Chapter 3, N. Nisan, T. Roughgarden, E.
Tardos, and V. Vazirani eds., Algorithmic Game Theory, 2007.

"""
import numpy as np
from numba import jit


TOL_PIV = 1e-10


def lemke_howson(g, init_pivot=0, max_iter=10**6):
    try:
        N = g.N
    except:
        raise ValueError('input must be a 2-player NormalFormGame')
    if N != 2:
        raise NotImplementedError('Implemented only for 2-player games')

    payoff_matrices = tuple(g.players[i].payoff_array for i in range(N))
    nums_actions = g.nums_actions
    total_num = sum(nums_actions)

    if not (0 <= init_pivot < total_num):
        raise ValueError(
            '`init_pivot` must be an integer k such that 0 <= k < {0}'
            .format(total_num)
        )

    tableaux = tuple(
        np.empty((nums_actions[1-i], total_num+1)) for i in range(N)
    )
    bases = tuple(np.empty(nums_actions[1-i], dtype=int) for i in range(N))

    initialize_tableaux(payoff_matrices, tableaux, bases)
    lemke_howson_tbl(tableaux, bases, init_pivot, max_iter)
    NE = get_mixed_actions(tableaux, bases)

    return NE


@jit(nopython=True)
def initialize_tableaux(payoff_matrices, tableaux, bases):
    nums_actions = payoff_matrices[0].shape

    consts = np.zeros(2)  # To be added to payoffs if min <= 0
    for pl in range(2):
        min_ = payoff_matrices[pl].min()
        if min_ <= 0:
            consts[pl] = min_ * (-1) + 1

    for pl, (py_start, sl_start) in enumerate(zip((0, nums_actions[0]),
                                                  (nums_actions[0], 0))):
        for i in range(nums_actions[1-pl]):
            for j in range(nums_actions[pl]):
                tableaux[pl][i, py_start+j] = \
                    payoff_matrices[1-pl][i, j] + consts[1-pl]
            for j in range(nums_actions[1-pl]):
                if j == i:
                    tableaux[pl][i, sl_start+j] = 1
                else:
                    tableaux[pl][i, sl_start+j] = 0
            tableaux[pl][i, -1] = 1

        for i in range(nums_actions[1-pl]):
            bases[pl][i] = sl_start + i

    return tableaux, bases


@jit(nopython=True)
def min_ratio_test(tableau, pivot):
    nrows = tableau.shape[0]

    row_min = 0
    while tableau[row_min, pivot] < TOL_PIV:  # Treated as nonpositive
        row_min += 1

    for i in range(row_min+1, nrows):
        if tableau[i, pivot] <= 0:
            continue
        if tableau[i, -1] * tableau[row_min, pivot] < \
           tableau[row_min, -1] * tableau[i, pivot]:
                row_min = i

    return row_min


@jit(nopython=True)
def pivoting(tableau, pivot, pivot_row):
    """
    Perform a pivoting step.

    Modify `tableau` in place (and return its view).

    """
    nrows, ncols = tableau.shape

    pivot_elt = tableau[pivot_row, pivot]
    for j in range(ncols):
        tableau[pivot_row, j] /= pivot_elt

    for i in range(nrows):
        if i == pivot_row:
            continue
        multiplier = tableau[i, pivot]
        if multiplier == 0:
            continue
        for j in range(ncols):
            tableau[i, j] -= tableau[pivot_row, j] * multiplier

    return tableau


@jit(nopython=True)
def lemke_howson_tbl(tableaux, bases, init_pivot, max_iter):
    init_player = 0
    for k in bases[0]:
        if k == init_pivot:
            init_player = 1
            break
    pls = [init_player, 1 - init_player]

    pivot = init_pivot

    success = False
    num_iter = 0

    while True:
        for pl in pls:
            # Determine the leaving variable
            row_min = min_ratio_test(tableaux[pl], pivot)

            # Pivoting step: modify tableau in place
            pivoting(tableaux[pl], pivot, row_min)

            # Update the basic variables and the pivot
            bases[pl][row_min], pivot = pivot, bases[pl][row_min]

            num_iter += 1

            if pivot == init_pivot:
                success = True
                break
            if num_iter >= max_iter:
                break
        else:
            continue
        break

    return success, num_iter


@jit(nopython=True)
def get_mixed_actions(tableaux, bases):
    nums_actions = tableaux[1].shape[0], tableaux[0].shape[0]
    num = nums_actions[0] + nums_actions[1]
    out = np.zeros(num)

    for pl, (start, stop) in enumerate(zip((0, nums_actions[0]),
                                           (nums_actions[0], num))):
        sum_ = 0.
        for i in range(nums_actions[1-pl]):
            k = bases[pl][i]
            if start <= k < stop:
                out[k] = tableaux[pl][i, -1]
                sum_ += tableaux[pl][i, -1]
        if sum_ != 0:
            out[start:stop] /= sum_

    return out[:nums_actions[0]], out[nums_actions[0]:]
