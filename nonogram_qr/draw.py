import os
import shutil
from tempfile import mkstemp
import itertools as it

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import ArtistAnimation


def draw_nonogram(row_rle, col_rle, shape=None, mat=None):
    '''
    Draw a Nonogram puzzle using matplotlib

    Parameters
    ----------

    row_rle: list, required
        List of lists or list of arrays, where each array is the run length
        encoded clues for a row of the nonogram.

    col_rle: list, required
        List of lists or list of arrays, where each array is the run length
        encoded clues for a column of the nonogram.

    shape: tuple or list of length 2 or None, optional, default: None
        Tuple of length 2 containing the size of the nonogram matrix.

    mat: np.ndarray or None, optional, default: None
        The matrix to be solved. Can be partially completed already. Unsolved
        positions should be represented by -1, solved filled positions by 1,
        and solved unfilled positions by 0. If supplied, this parameter 
        overrides the size parameter. NB: original matrix is copied and not
        altered in place.

    Returns
    -------

    ax: matplotlib.axes.Axes
        The axes onto which the Nonogram is plotted

    '''
    if mat is None and shape is None:
        raise ValueError(
            'must provide either a grid shape or a matrix')
    elif shape is None or mat is not None:
        shape = mat.shape

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis('off')
    plt.axis('equal')

    r, c = shape
    # draw the grid for the nonogram:
    for i in range(r + 1):
        ax.plot([0, c], [-i, -i], 'k-')
    for j in range(c + 1):
        ax.plot([j, j], [0, -r], 'k-')

    # draw the numbers onto the grid
    for i, row in enumerate(row_rle):
        for idx, val in enumerate(row[::-1]):
            ax.annotate(xy=(-idx - 0.5, -i - 0.5), s=val, ha='center', va='center')
    for j, col in enumerate(col_rle):
        for idx, val in enumerate(col[::-1]):
            ax.annotate(xy=(j + 0.5, idx + 0.5), s=val, ha='center', va='center')

    if mat is not None:
        for i, j in np.argwhere(mat == 1):
            ax.add_patch(Rectangle(xy=(j, -i - 1), width=1, height=1, color='k'))
    # adjust x and y limits
    lim_left = max([len(x) for x in row_rle + col_rle]) + 1
    lim_right = max(r, c) + 1
    ax.set_xlim(-lim_left, lim_right)
    ax.set_ylim(-lim_right, lim_left)
    return ax


def draw_solution_so_far(mat, ax):
    '''
    Draw nonogram solution including unsolved squares (in grey).
    '''
    r, c = mat.shape
    patches = []
    for i, j in it.product(range(r), range(c)):
        c = ['#ffffff', '#000000', '#999999'][mat[i, j]]
        patches.append(ax.add_patch(Rectangle((j + 1, -i), 1, 1, color=c)))
    return patches


def draw_solver_progress(progress_mats,
                         display_with_ipython=True,
                         img_filename=None,
                         imagemagick_path=None,
                         imagemagick_extra_args=None):
    '''
    Produce a GIF of solution progress using matplotlib, requires imagemagick.

    Parameters
    ----------

    progress_mats: list of 2D arrays, required
        List of matrices showing progression of brute force solver. Produced by
        solve_nonogram when return_iterations is set to True

    display_with_ipython: bool, optional, default: True
        Whether or not to display image using IPython.display.Image (for jupyter
        notebook or console)

    img_filename: str or None, optional, default=None
        path to output GIF file. If not specified, a temporary file will be
        created for display.
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.autoscale()
    imgs = [draw_solution_so_far(mat, ax) for mat in progress_mats]
    anim = ArtistAnimation(fig, imgs, interval=500, repeat_delay=1000)
    plt.axis('square')
    plt.axis('off')
    if img_filename is None:
        handle, img_filename = mkstemp(suffix='.gif')
        os.close(handle)
    if imagemagick_path:
        plt.rcParams["animation.convert_path"] = imagemagick_path
    anim.save(img_filename, writer='imagemagick', extra_args=imagemagick_extra_args)
    plt.close()
    if display_with_ipython:
        from IPython.display import Image, display
        shutil.copy2(img_filename, img_filename + '.png')
        display(Image(filename=img_filename + '.png'))
