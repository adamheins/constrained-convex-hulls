"""Generate constrained convex hulls as an SVG."""

import argparse

import drawsvg as draw
import numpy as np

import constrained_convex_hulls as cch


SVG_WIDTH = 700
SVG_SHAPE_HEIGHT = 300
SVG_HEIGHT = 3 * SVG_SHAPE_HEIGHT


def main():
    np.set_printoptions(suppress=True, precision=4)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=11,  # picked because I think the shape looks interesting
        help="Random seed to generate the shape.",
    )
    parser.add_argument(
        "-l",
        "--lower-bound",
        type=float,
        default=0.05,
        help="Lower bounds on the convex hull weights.",
    )
    parser.add_argument(
        "-u",
        "--upper-bound",
        type=float,
        default=0.6,
        help="Upper bounds on the convex hull weights.",
    )
    parser.add_argument(
        "-v",
        "--max-vertices",
        type=int,
        default=6,
        help="Maximum number of vertices of the original shape.",
    )
    args = parser.parse_args()

    assert 0 <= args.lower_bound <= args.upper_bound <= 1

    np.random.seed(args.seed)

    shape = np.array([SVG_WIDTH, SVG_SHAPE_HEIGHT])
    V = cch.random_vertices(args.max_vertices) * shape + shape / 2
    n = V.shape[0]

    l = args.lower_bound * np.ones(n)
    u = args.upper_bound * np.ones(n)

    V_l = cch.constrain_vertex_weights(V, l=l, u=np.ones_like(u))
    V_u = cch.constrain_vertex_weights(V, l=np.zeros_like(l), u=u)
    V_lu = cch.constrain_vertex_weights(V, l=l, u=u)

    hull1 = cch.svg_poly(V, cch.SVG_BLUE)
    hull2 = cch.svg_poly(V, cch.SVG_BLUE, offset=(0, SVG_SHAPE_HEIGHT))
    hull3 = cch.svg_poly(V, cch.SVG_BLUE, offset=(0, 2 * SVG_SHAPE_HEIGHT))

    hull_l = cch.svg_poly(V_l, cch.SVG_RED)
    hull_u = cch.svg_poly(V_u, cch.SVG_RED, offset=(0, SVG_SHAPE_HEIGHT))
    hull_lu = cch.svg_poly(V_lu, cch.SVG_RED, offset=(0, 2 * SVG_SHAPE_HEIGHT))

    d = draw.Drawing(SVG_WIDTH, SVG_HEIGHT)
    d.extend([hull1, hull2, hull3, hull_l, hull_u, hull_lu])
    d.save_svg("cch.svg")


if __name__ == "__main__":
    main()
