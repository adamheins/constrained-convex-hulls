"""Utilities for constrained convex hulls."""
import numpy as np
import scipy


def conv(points):
    """Take the convex hull of the points, such that non-extreme points are
    removed.

    For points in two-dimensions, the result is guaranteed to be in
    counter-clockwise order.

    Parameters
    ----------
    points : np.ndarray,
        The array of points, one per row.

    Returns
    -------
    : np.ndarray
        The array of vertices.
    """
    hull = scipy.spatial.ConvexHull(points)
    return points[hull.vertices]


def square_vertices(s):
    """Generate vertices of a square centered on the origin.

    Parameters
    ----------
    s : float, positive
        The side length.

    Returns
    -------
    : np.ndarray, shape (4, 2)
        The vertices of the square, one per row.
    """
    r = s / 2
    return np.array([[r, r], [-r, r], [-r, -r], [r, -r]])


def random_vertices(n, d=2):
    """Return the vertices of a convex shape with at most n vertices in d
    dimensions.

    If ``d==2``, vertices are guaranteed to be in counter-clockwise order.

    Parameters
    ----------
    n : int
        Maximum number of vertices.
    d : int
        Dimension of the space.

    Returns
    -------
    : np.ndarray
        Array of vertices, where each row is a vertex.
    """
    V = np.random.random((n, d)) - 0.5
    return conv(V)


def scale_and_center_vertices(V, shape):
    """Scale and center a set of random vertices.

    Parameters
    ----------
    V : np.ndarray, shape (n, d)
        The array of vertices.
    shape : np.ndarray, shape (d,)
        The shape to transform the vertices into.

    Returns
    -------
    : np.ndarray, shape (n, d)
        The transformed vertices.
    """
    xy_min = np.min(V, axis=0)
    xy_max = np.max(V, axis=0)
    xy_mid = 0.5 * (xy_min + xy_max)
    return shape * (V - xy_mid + 0.5)


def vertex_match(v, V):
    """Check if a vertex lies in a set.

    Parameters
    ----------
    v : np.ndarray, shape (d,)
        The vertex to check.
    V : np.ndarray, shape (n, d)
        The set of vertices to check against.

    Returns
    -------
    : bool
        True if ``v`` equals at least one row of ``V``, False otherwise.
    """
    return np.any(np.all(np.isclose(V - v, 0), axis=1))


def vertices_match(V1, V2):
    """Check if two sets of vertices match, regardless of order.

    Parameters
    ----------
    V1 : np.ndarray, shape (n, d)
        The first set of vertices, one per row.
    V2 : np.ndarray, shape (n, d)
        The second set of vertices.

    Returns
    -------
    : bool
        True if the two arrays contain the same vertices, False otherwise.
    """
    assert V1.shape == V2.shape
    D = scipy.spatial.distance.cdist(V1, V2)
    d = np.min(D, axis=1)
    return np.allclose(d, 0)
