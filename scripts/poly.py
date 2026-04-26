"""Generate examples of polygons as an SVG."""

import drawsvg as draw
import numpy as np

import constrained_convex_hull as cch


SVG_WIDTH = 700
SVG_HEIGHT = 300

SVG_BLUE = "rgb(47, 103, 177)"
SVG_RED = "rgb(191, 44, 35)"


def main():
    np.random.seed(11)

    # scaling and offset are fairly arbitrary; just designed to make a nice
    # looking figure
    shape = np.array([SVG_WIDTH, SVG_HEIGHT])
    V_random = cch.random_vertices(6) * shape / 1.75 + (500, 175)
    V_square = cch.square_verts(200) + (150, 150)

    d = draw.Drawing(SVG_WIDTH, SVG_HEIGHT)
    d.append(cch.svg_poly(V_square, SVG_RED))
    d.append(cch.svg_poly(V_random, SVG_BLUE))
    d.extend(cch.svg_points(V_square))
    d.extend(cch.svg_points(V_random))
    d.save_svg("polygons.svg")


if __name__ == "__main__":
    main()
