import foronoi

from foronoi.contrib import ConcavePolygon

points = [(0.1, 0.2), (0.1, 0.6), (0.75, 0.1)]

poly_nodes = [(0, 0), (0, 1), (1, 1), (1, 0)]
# poly_nodes.insert(3, (7, 4))
poly_nodes.insert(3, (0.6, 0.5))

# v = foronoi.Voronoi(foronoi.Polygon(poly_nodes))
v = foronoi.Voronoi(ConcavePolygon(poly_nodes))

v.create_diagram(points=points)
foronoi.Visualizer(v).plot_sites(init_order_names=True).plot_edges(
    show_labels=False
).plot_vertices().show()

# need to insert new vertices (where HalfEdge meets polygon) correctly (edge we detect intersection
# with needs to return edges it interacts with/indices etc.)

# if edge intersects more than once, need to make more than one new vertex & edge -> bit tricky
