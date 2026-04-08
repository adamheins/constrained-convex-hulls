import matplotlib.pyplot as plt
import numpy as np
import scipy
import cdd

import IPython


def square_verts(s):
    r = s / 2
    return np.array([[r, r], [-r, r], [-r, -r], [r, -r]])


def random_vertices(n):
    """Return the vertices of a convex shape with at most n vertices.

    Vertices are guaranteed to be in counter-clockwise order.

    Parameters
    ----------
    n : int
        Maximum number of vertices.

    Returns
    -------
    : np.ndarray
        Array of vertices, where each row is a vertex.
    """
    V = np.random.random((n, 2)) - 0.5
    return wind_ccw(V)


def wind_ccw(V):
    """Wind vertices in counter-clockwise order. Non-extreme points are also
    removed.

    This is typically needed for rendering.

    Parameters
    ----------
    V : np.ndarray
        The array of vertices, one per row.

    Returns
    -------
    : np.ndarray
        The array of vertices in counter-clockwise order.
    """
    hull = scipy.spatial.ConvexHull(V)
    return V[hull.vertices]


def constrain_vertex_weights(V, l, u):
    """Generate new vertices with constraints on hull weights."""
    nv = V.shape[0]

    # construct H-rep of weights
    # Ax + b >= 0
    # Ax + b == 0
    A = np.vstack((np.eye(nv), np.eye(nv), -np.eye(nv), np.ones((1, nv))))
    b = np.concatenate((np.zeros(nv), -l, u, [-1]))
    M = np.hstack((b[:, None], A))
    lin_set = [M.shape[0] - 1]  # last row is the only equality

    mat = cdd.matrix_from_array(
        array=M, lin_set=lin_set, rep_type=cdd.RepType.INEQUALITY
    )
    cdd.matrix_canonicalize(mat)
    poly = cdd.polyhedron_from_matrix(mat)

    # convert to V-rep, where rows are the new vertices
    Vw = np.array(cdd.copy_generators(poly).array)[:, 1:]

    # combine to make the new polyhedron
    V2 = Vw @ V
    return wind_ccw(V2)


def main():
    # V = square_verts(s=1)
    V = random_vertices(6)
    nv = V.shape[0]

    l = 0.01 * np.ones(nv)
    u = 0.75 * np.ones(nv)

    V2 = constrain_vertex_weights(V, l, u)

    fig, ax = plt.subplots()
    plt.fill(V[:, 0], V[:, 1], color="blue")
    plt.fill(V2[:, 0], V2[:, 1], color="red")
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
