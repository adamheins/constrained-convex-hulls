"""Tools for generating SVGs."""

import drawsvg as draw
import numpy as np


SVG_BLUE = "#2f67b1"
SVG_RED = "#bf2c23"
SVG_GREEN = "#45a051"
SVG_YELLOW = "#f2e021"


def svg_poly(vertices, fill, offset=None, **kwargs):
    """Make the SVG representation of a polygon.

    Parameters
    ----------
    vertices : np.ndarray, shape (nv, 2)
        The vertices of the polygon.
    fill : str
        The fill color of the shape.
    offset : np.ndarray, shape (2,), or None
        Offset to add to all vertices.
    kwargs : dict
        Additional keyword arguments passed to ``drawsvg.Lines``.

    Returns
    -------
    : drawsvg.Lines
        The SVG polygon object.
    """
    if offset is None:
        offset = np.zeros(2)
    V = vertices + offset

    default_kwargs = {"stroke": "black", "fill": fill, "stroke-width": 3, "close": True}
    default_kwargs.update(kwargs)

    return draw.Lines(
        *V.flatten().tolist(),
        **default_kwargs,
    )


def svg_points(points, radius=5):
    """Make a set of SVG points.

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


def svg_line(start, end, **kwargs):
    """Make a single SVG line.

    Parameters
    ----------
    start : iterable, shape (2,)
        Start point.
    end : iterable, shape (2,)
        End point.
    kwargs : dict
        Additional keyword arguments passed to ``drawsvg.Line``.

    Returns
    -------
    : drawsvg.Line
        The SVG line object.
    """

    default_kwargs = {"stroke": "black", "stroke-width": 3}
    default_kwargs.update(kwargs)

    return draw.Line(
        start[0],
        start[1],
        end[0],
        end[1],
        **default_kwargs,
    )
