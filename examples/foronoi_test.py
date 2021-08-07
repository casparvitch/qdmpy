import foronoi

from foronoi.contrib import ConcavePolygon

points = [(2.5, 2.5), (4, 7.5), (7.5, 2.5), (6, 7.5), (4, 4), (3, 3), (6, 2)]

poly_nodes = [(2.5, 10), (5, 10), (10, 5), (10, 2.5), (5, 0), (2.5, 0), (0, 2.5), (0, 5)]
# poly_nodes.insert(3, (7, 4))
poly_nodes.insert(3, (4, 3))

# v = foronoi.Voronoi(foronoi.Polygon(poly_nodes))
v = foronoi.Voronoi(ConcavePolygon(poly_nodes))

v.create_diagram(points=points)
foronoi.Visualizer(v).plot_sites(init_order_names=False).plot_edges(
    show_labels=False
).plot_vertices().show()
