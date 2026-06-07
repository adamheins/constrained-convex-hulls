"""Core tools for constrained convex hulls."""

import cdd
import numpy as np

from .util import conv


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
    V = np.array(V)
    l = np.array(l)
    u = np.array(u)

    assert np.all(l >= 0), "Lower bounds must be non-negative."
    assert np.all(u <= 1), "Upper bounds must be at most one."
    assert np.all(l <= u), "Lower bounds must be less than or equal to upper bounds."

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

    A = cdd.copy_generators(poly).array
    if len(A) == 0:
        raise ValueError("Constraints are infeasible.")

    # convert to V-rep, where rows are the new vertices
    Vw = np.array(A)[:, 1:]

    # combine to make the new polyhedron
    V2 = Vw @ V
    return conv(V2)


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
    mat = cdd.matrix_from_array(array=M, rep_type=cdd.RepType.GENERATOR)
    poly = cdd.polyhedron_from_matrix(mat)

    # convert to H-rep: Ax + b >= 0
    bA = np.array(cdd.copy_inequalities(poly).array)

    # remove width from b, but we need to ensure each row of A is unit-length
    # for this to produce correct results
    norms = np.linalg.norm(bA[:, 1:], axis=1)
    bA /= norms[:, None]
    bA[:, 0] -= width

    # convert back to V-rep
    mat = cdd.matrix_from_array(array=bA, rep_type=cdd.RepType.INEQUALITY)
    poly = cdd.polyhedron_from_matrix(mat)
    V_in = np.array(cdd.copy_generators(poly).array)[:, 1:]
    return conv(V_in)


def vertex_adjacency(V):
    """Determine the neighbours of each vertex.

    Parameters
    ----------
    V : np.ndarray, shape (n, d)
        The array of vertices, one per row.

    Returns
    -------
    : Sequence[Set[int]]
        A sequence containing the indices of the neighbours of each vertex.
    """
    n = vertices.shape[0]
    M = np.hstack((np.ones((n, 1)), vertices))
    mat = cdd.matrix_from_array(array=M, rep_type=cdd.RepType.GENERATOR)
    return cdd.matrix_adjacency(mat)


def analytic_lower_bounded_vertices(V, l):
    """Compute the lower-bounded convex hull analytically.

    Parameters
    ----------
    V : np.ndarray
        The array of vertices, one per row.
    l : np.ndarray
        Lower bounds on the vertex weights.

    Returns
    -------
    : np.ndarray
        The new constrained vertices.
    """
    nv = V.shape[0]
    V_new = []
    for i in range(nv):
        v_new = V[i, :] + np.sum(l[:, None] * (V - V[i, :]), axis=0)
        V_new.append(v_new)
    return np.array(V_new)


def analytic_upper_bounded_vertices(V, u):
    """Compute the upper-bounded convex hull analytically.

    Note that this is only valid if for each vertex V[i, :] and each of its
    neighbours V[j, :], the corresponding bounds satisfy u[i] + u[j] >= 1.

    Parameters
    ----------
    V : np.ndarray
        The array of vertices, one per row.
    u : np.ndarray
        Upper bounds on the vertex weights.

    Returns
    -------
    : np.ndarray
        The new constrained vertices.
    """
    nv = V.shape[0]
    adj = vertex_adjacency(V)
    V_new = []
    for i in range(nv):
        for j in adj[i]:
            v_new = V[j, :] + u[i] * (V[i, :] - V[j, :])
            V_new.append(v_new)
    V_new = np.array(V_new)
    return conv(V_new)


def random_lower_bounds(n):
    """Generate valid random lower bounds for the vertex weights.

    The bounds must be non-negative and sum to at most one.

    Parameters
    ----------
    n : int, positive
        The number of bounds.

    Returns
    -------
    : np.ndarray, shape (n,)
        The random lower bounds.
    """
    # in principle we can have some weights higher than 1 / n while still
    # having all of them summing to at most 1, but this is much faster
    return np.random.random(n) / n


def random_upper_bounds(n):
    """Generate valid random upper bounds for the vertex weights.

    The bounds must be non-negative and sum to at least one.

    Parameters
    ----------
    n : int, positive
        The number of bounds.

    Returns
    -------
    : np.ndarray, shape (n,)
        The random upper bounds.
    """
    # we enforce the constraint that each bound must be at least 0.5, which
    # simplifies the analytical construction
    u = 0.5 * (1 + np.random.random(n))
    while np.sum(u) < 1:
        u = 0.5 * (1 + np.random.random(n))
    return u
