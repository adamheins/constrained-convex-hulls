"""Generate lower- or upper-bounded convex hulls analytically as SVGs."""

import numpy as np
import constrained_convex_hull as cch


def test_analytical_bounds(n, d):
    V = cch.random_vertices(n, d)
    n = V.shape[0]

    l = cch.random_lower_bounds(n)
    V_l_act = cch.analytic_lower_bounded_vertices(V, l=l)
    V_l_exp = cch.constrain_vertex_weights(V, l=l, u=np.ones_like(l))
    assert cch.vertices_match(V_l_act, V_l_exp), "Incorrect lower-bounded hull"

    u = cch.random_upper_bounds(n)
    V_u_act = cch.analytic_upper_bounded_vertices(V, u=u)
    V_u_exp = cch.constrain_vertex_weights(V, l=np.zeros_like(u), u=u)
    assert cch.vertices_match(V_u_act, V_u_exp), "Incorrect upper-bounded hull"


def main():
    np.set_printoptions(suppress=True, precision=4)

    np.random.seed(11)

    # random tests in dimensions 2, 3, 4
    for _ in range(100):
        test_analytical_bounds(n=10, d=2)
    for _ in range(100):
        test_analytical_bounds(n=10, d=3)
    for _ in range(100):
        test_analytical_bounds(n=10, d=4)


if __name__ == "__main__":
    main()
