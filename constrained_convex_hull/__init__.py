import argparse

import cdd
import drawsvg as draw
import numpy as np
import scipy


def square_verts(s):
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


def svg_poly(vertices, fill, offset=None):
    """Make the SVG representation of a closed polygon.

    Parameters
    ----------
    vertices : np.ndarray, shape (nv, 2)
        The vertices of the polygon.
    fill : str
        The fill color of the shape.
    offset : np.ndarray, shape (2,), or None
        Offset to add to all vertices.

    Returns
    -------
    : drawsvg.Lines
        The SVG polygon object.
    """
    if offset is None:
        offset = np.zeros(2)
    V = vertices + offset

    return draw.Lines(
        *V.flatten().tolist(),
        stroke="black",
        fill=fill,
        stroke_width=3,
        close=True,
    )


def svg_points(points, radius=5):
    return [draw.Circle(x, y, radius) for x, y in points]


