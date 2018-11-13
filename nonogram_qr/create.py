import numpy as np
import qrcode

from .solve import solve_nonogram


def qr_matrix(data):
    '''
    Produce a QR code matrix using the package qrcode.
    '''
    qr = qrcode.QRCode(version=None,
                       box_size=1,
                       border=1,
                       error_correction=qrcode.ERROR_CORRECT_L)
    qr.add_data(data)
    qr.make(fit=True)
    return np.asarray(qr.get_matrix(), dtype = int)


def _rle(matrix):
    '''
    Run length encode rows of a matrix
    '''
    # find run start and ends
    d = np.diff(matrix)
    row, start_pos = np.where(d > 0)
    _, end_pos = np.where(d < 0)
    # find run lengths
    run_lengths = end_pos - start_pos
    # split runs from different rows into separate arrays
    split_on = np.cumsum(np.bincount(row - 1))[:-1]
    return np.split(run_lengths, split_on)


def run_length_encode(matrix):
    '''
    Produce row and column run length encodings of matrix
    (to create nonogram clues)
    '''
    rle_row = _rle(matrix)
    rle_col = _rle(matrix.T)
    return rle_row, rle_col


def nonogram_qr(data, fill_ambiguous=True):
    '''
    create a QR code from a string, then convert it into a Nonogram puzzle

    Parameters
    ----------

    data: str, required
        The string to be encoded in the QR code

    fill_ambiguous: bool, optional, default: True
        If True, an attempt is made to solve the Nonogram using a brute
        force method. If the matrix cannot be solved, any ambiguous
        squares are filled to produce an unambiguous solution.

    Returns
    -------

    shape: tuple, shape (2,)
        The shape of the Nonogram/QR code matrix

    row_rle: list of arrays, shape (N,)
        The run length encoded clues for the rows of the nonogram

    col_rle: list of arrays, shape (M,)
        The run length encoded clues for the columns of the nonogram

    mat: np.ndarray, shape (N, M)
        The initial matrix to be solved. If the solution is unambiguous
        or fill_ambiguous is set to False, this is a matrix filled with
        -1s. Otherwise, it may contain a partial solution.
    '''
    qr = qr_matrix(data)
    qr_unpadded = qr[1:-1, 1:-1]
    row_rle, col_rle = run_length_encode(qr)
    shape = np.array(qr.shape) - 2
    mat = np.full(shape, -1, dtype=np.int)
    if fill_ambiguous:
        solved, ambig = solve_nonogram(row_rle, col_rle,
                                       mat=mat,
                                       return_iterations=False)
        if ambig:
            mat[solved == -1] = qr_unpadded[solved == -1]
            mat[mat == 0] = -1
    return tuple(shape), row_rle, col_rle, mat
