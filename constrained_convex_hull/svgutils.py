"""Tools for generating SVGs."""
import drawsvg as draw
import numpy as np


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
    """Make SVG points.

    Parameters
    ----------
    points : list
        List of (x, y) points.
    radius : int
        Radius of each point.

    Returns
    -------
    : list[drawsvg.Circle]
        The SVG points.
    """
    return [draw.Circle(x, y, radius) for x, y in points]
