"""Example of a uniformly-padded convex hull."""

import argparse

import drawsvg as draw
import numpy as np

import constrained_convex_hull as cch


SVG_WIDTH = 700
SVG_HEIGHT = 300


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
        "-w",
        "--width",
        type=float,
        default=50,
        help="Width of the uniform padding.",
    )
    parser.add_argument(
        "-v",
        "--max-vertices",
        type=int,
        default=6,
        help="Maximum number of vertices of the original shape.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    shape = np.array([SVG_WIDTH, SVG_HEIGHT])
    V = cch.random_vertices(args.max_vertices) * shape + shape / 2
    nv = V.shape[0]

    V_in = cch.inset_convex_hull_uniform(vertices=V, width=args.width)

    hull = cch.svg_poly(V, cch.SVG_BLUE)
    hull_in = cch.svg_poly(V_in, cch.SVG_RED)

    d = draw.Drawing(SVG_WIDTH, SVG_HEIGHT)
    d.extend([hull, hull_in])
    d.save_svg("uniform_padding.svg")


if __name__ == "__main__":
    main()
