from functools import reduce
from warnings import warn

import numpy as np

def unique_permutations(seq):
    '''
    Permute a sequence to get all unique orderings
    '''
    i_indices = range(len(seq) - 1, -1, -1)
    k_indices = i_indices[1:]
    seq = sorted(seq)
    while True:
        yield seq
        for k in k_indices:
            if seq[k] < seq[k + 1]:
                break
        else:
            # once sequence is completely reversed, we are done
            return
        k_val = seq[k]
        for i in i_indices:
            if k_val < seq[i]:
                break
        (seq[k], seq[i]) = (seq[i], seq[k])
        seq[k + 1:] = seq[-1:k:-1]


def permute_partitions(partitions, n_partitions):
    '''
    Padded integer partitions to correct length and permute numbers to get
    all unique orderings
    '''
    partitions = partitions + [0] * (n_partitions - len(partitions))
    for order in unique_permutations(partitions):
        yield np.array(order)


def integer_partition(n, n_partitions):
    '''
    Split a number n into n_partitions which can be used as the sizes of the
    spaces between filled regions in a row or column.
    '''
    # adapted from http://jeromekelleher.net/generating-integer-partitions.html
    a = [0 for i in range(n + 1)]
    k = 1
    a[1] = n
    while k != 0:
        x = a[k - 1] + 1
        y = a[k] - 1
        k -= 1
        while x <= y and k < n_partitions - 1:
            a[k] = x
            y -= x
            k += 1
        a[k] = x + y
        yield from permute_partitions(a[:k + 1], n_partitions)


def rle_to_binary_arr(row_size, row_rle, space_rle):
    '''
    Convert a run length encoding of the filled and space sizes to a
    binary array (filled positions represented by ones, unfilled by zeros)
    '''
    if not len(row_rle):
        return np.empty(row_size, dtype=np.int)
    row_rle = iter(row_rle)
    arr = np.empty(row_size, dtype=np.int)
    pos = 0
    for i in space_rle[:-1]:
        arr[pos: pos + i] = 0
        pos += i
        j = next(row_rle)
        arr[pos: pos + j] = 1
        pos += j
    if space_rle[-1]:
        arr[-space_rle[-1]:] = 0
    return arr


def min_size(row_rle):
    '''
    Calculates the minimum length of binary array that could fulfil row_rle
    (i.e. all spaces between filled positions length one)
    '''
    return row_rle.sum() + len(row_rle) - 1


def all_row_permutations(row_size, row_rle):
    '''
    Generate all possible solutions for a row given a set of clues (row_rle)
    and the size of the row in the nonogram grid.
    '''
    min_row_size = min_size(row_rle)
    extra_space = row_size - min_row_size
    init_space_rle = np.array([0] + [1] * (len(row_rle) - 1) + [0])
    if extra_space == 0:
        # There is only one possible solution with all gaps of one:
        return [rle_to_binary_arr(row_size, row_rle, init_space_rle)]
    elif extra_space < 0:
        raise ValueError(
            'Matrix size is not big enough for row_rle {}'.format(row_rle))
    row_perm = []
    for p in integer_partition(extra_space, init_space_rle.size):
        space_rle = init_space_rle + p
        row_perm.append(
            rle_to_binary_arr(row_size, row_rle, space_rle)
        )
    return row_perm


def filter_perm(row, row_permutations):
    '''
    Work out which possible permutations of a row are still possible solutions
    given what we already know about the row.
    '''
    mask = row != -1
    row_masked = row[mask]
    if row_masked.size:
        still_possible = []
        for perm in row_permutations:
            if np.all(row_masked == perm[mask]):
                still_possible.append(perm)
        return still_possible
    else:
        return row_permutations


def update_row(row, row_permutations):
    '''
    Find the intersection of filled and unfilled squares in all row permutations
    to identify solved positions, and update row
    '''
    filled = reduce(np.logical_and, row_permutations).astype(bool)
    unfilled = reduce(np.logical_and,
                      [1 - x for x in row_permutations]).astype(bool)
    row[filled] = 1
    row[unfilled] = 0
    return row


def solve_round(mat, row_perm):
    '''
    Perform one iteration of solving by filtering the existing row_permutations
    and identifying and new solved positions.
    '''
    filtered_row_perm = []
    for i, row_p in enumerate(row_perm):
        filtered = filter_perm(mat[i], row_p)
        solved = update_row(mat[i], filtered)
        filtered_row_perm.append(filtered)
        mat[i] = solved
    return mat, filtered_row_perm


def solve_nonogram(row_rle, col_rle, shape=None, mat=None,
                   return_iterations=False):
    '''
    Brute force solve a nonogram using an iterative approach.

    Parameters
    ----------
    row_rle: list, required
        List of lists or list or arrays, where each array is the run length
        encoded clues for a row of the nonogram.

    col_rle: list, required
        List of lists or list or arrays, where each array is the run length
        encoded clues for a column of the nonogram.

    shape: tuple or list of length 2 or None, optional, default: None
        Tuple of length 2 containing the size of the nonogram matrix.

    mat: np.ndarray or None, optional, default: None
        The matrix to be solved. Can be partially completed already. Unsolved
        positions should be represented by -1, solved filled positions by 1,
        and solved unfilled positions by 0. If supplied, this parameter 
        overrides the size parameter. NB: original matrix is copied and not
        altered in place.

    return_iterations: bool, optional, default: False
        Whether or not to return the list of partially solved matrices which
        are produced by each iteration of the solver. These can be used to
        generate a animation showing the solving process.

    Returns
    -------

    mat: np.ndarray
        The solved Nonogram grid. Filled positions are represented by 1,
        unfilled by 0, and ambiguous unsolvable positions by -1
    ambig: bool
        Whether the Nonogram is ambiguous and only partially solvable,
        or not.
    iters: list of arrays
        Only returned if return_iterations is True. A list of partially
        solved matrices showing the progress of the solver.
    '''
    if mat is None and shape is None:
        raise ValueError(
            'must provide either a matrix size or a matrix to fill')
    elif mat is None:
        mat = np.full(shape, -1, dtype=np.int)
    else:
        mat = mat.copy()
        shape = mat.shape
    row_perm = [all_row_permutations(shape[0], row) for row in row_rle]
    col_perm = [all_row_permutations(shape[1], col) for col in col_rle]
    n = 1
    if return_iterations:
        iters = []
        iters.append(mat)
    while np.any(mat == -1):
        cur_mat = mat.copy()
        cur_mat, row_perm = solve_round(cur_mat, row_perm)
        if return_iterations:
            iters.append(cur_mat)
        cur_mat, col_perm = solve_round(cur_mat.transpose(), col_perm)
        cur_mat = cur_mat.transpose()
        if return_iterations:
            iters.append(cur_mat)
        if np.all(cur_mat == mat):
            warn('There are multiple ambiguous solutions', UserWarning)
            ambig = True
            break
        mat = cur_mat
        n += 1
    else:
        ambig = False
    if return_iterations:
        return mat, ambig, iters
    else:
        return mat, ambig
