"""Generate lower- or upper-bounded convex hulls analytically as SVGs."""

import drawsvg as draw
import numpy as np

import constrained_convex_hull as cch


SVG_WIDTH = 700
SVG_HEIGHT = 300


def main():
    np.set_printoptions(suppress=True, precision=4)

    np.random.seed(11)

    # V = cch.random_vertices(6, d=2)
    shape = np.array([SVG_WIDTH, SVG_HEIGHT])
    V = cch.random_vertices(6)
    V = cch.scale_and_center_vertices(V, shape)
    n = V.shape[0]

    l = np.zeros(n)
    l[0] = 0.2
    V_l_1 = cch.analytic_lower_bounded_vertices(V, l)

    u = np.ones(n)
    u[0] = 0.6
    V_u_1 = cch.analytic_upper_bounded_vertices(V, u)

    background = draw.Rectangle(0, 0, SVG_WIDTH, SVG_HEIGHT, fill="#bbbbbb", rx=20)
    hull = cch.svg_poly(V, cch.SVG_BLUE)

    # lower-bounded hull
    hull_l_1 = cch.svg_poly(V_l_1, cch.SVG_RED)
    d_l_1 = draw.Drawing(SVG_WIDTH, SVG_HEIGHT)
    d_l_1.extend([background, hull, hull_l_1])

    # draw lines to each vertex
    # green part is proportional to (1 - l), yellow part to l
    for i in range(1, n):
        v_mid = V[0, :] + (1 - l[0]) * (V[i, :] - V[0, :])
        d_l_1.append(cch.svg_line(V[0, :], v_mid, stroke=cch.SVG_GREEN))
        d_l_1.append(cch.svg_line(v_mid, V[i, :], stroke=cch.SVG_YELLOW))

    d_l_1.extend(cch.svg_points(V))
    d_l_1.extend(cch.svg_points(V_l_1))
    d_l_1.save_svg("one_lower_bound.svg")

    # upper-bounded hull
    hull_u_1 = cch.svg_poly(V_u_1, cch.SVG_RED, close=False)
    d_u_1 = draw.Drawing(SVG_WIDTH, SVG_HEIGHT)
    d_u_1.extend([background, hull, hull_u_1])

    # draw the new face separately so we can highlight it with dashing
    d_u_1.append(cch.svg_line(V_u_1[-1, :], V_u_1[0, :], stroke_dasharray=10))

    # draw lines to each vertex
    # green part is proportional to (1 - u), yellow part to u
    for i in [-1, 1]:
        v_mid = V[0, :] + (1 - u[0]) * (V[i, :] - V[0, :])
        d_u_1.append(cch.svg_line(V[0, :], v_mid, stroke=cch.SVG_GREEN))
        d_u_1.append(cch.svg_line(v_mid, V[i, :], stroke=cch.SVG_YELLOW))

    d_u_1.extend(cch.svg_points(V))
    d_u_1.extend(cch.svg_points(V_u_1))
    d_u_1.save_svg("one_upper_bound.svg")


if __name__ == "__main__":
    main()
