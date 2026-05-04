"""Core tools for constrained convex hulls."""
import cdd
import numpy as np
import scipy


def square_vertices(s):
    """Generate vertices of a square of side length ``s``, centered at zero."""
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
    """Generate new vertices with constraints on hull weights.

    Parameters
    ----------
    V : np.ndarray
        The array of vertices, one per row.
    l : np.ndarray
        Lower bounds on the vertex weights.
    u : np.ndarray
        Upper bounds on the vertex weights.

    Returns
    -------
    : np.ndarray
        The new constrained vertices.
    """
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


def inset_convex_hull_uniform(vertices, width):
    """Apply uniform padding to a convex hull.

    Parameters
    ----------
    vertices : np.ndarray, shape (n, 2)
        The vertices of the convex hull.
    width : float
        Width of the padding to add.

    Returns
    -------
    : np.ndarray
        The vertices of the padded convex hull.
    """
    # construct the polygon
    n = vertices.shape[0]
    M = np.hstack((np.ones((n, 1)), vertices))
    mat = cdd.matrix_from_array(
        array=M, rep_type=cdd.RepType.GENERATOR
    )
    poly = cdd.polyhedron_from_matrix(mat)

    # convert to H-rep: Ax + b >= 0
    bA = np.array(cdd.copy_inequalities(poly).array)

    # remove width from b, but we need to ensure each row of A is unit-length
    # for this to produce correct results
    norms = np.linalg.norm(bA[:, 1:], axis=1)
    bA /= norms[:, None]
    bA[:, 0] -= width

    # convert back to V-rep
    mat = cdd.matrix_from_array(
        array=bA, rep_type=cdd.RepType.INEQUALITY
    )
    poly = cdd.polyhedron_from_matrix(mat)
    V_in = np.array(cdd.copy_generators(poly).array)[:, 1:]
    return wind_ccw(V_in)
