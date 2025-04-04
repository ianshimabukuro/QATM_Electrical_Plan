import networkx as nx
import itertools

def dots_to_hanan(all_pixel_coords,nms_keep_indices, scale):
    dots_array = [ (int(all_pixel_coords[1][i]), int(all_pixel_coords[0][i])) for i in nms_keep_indices ]  
    
    x_coords = sorted(set(scale*x for x, y in dots_array))
    y_coords = sorted(set(scale*y for x, y in dots_array))

    # === Step 3: Generate Hanan grid points ===
    hanan_points = list(itertools.product(x_coords, y_coords))  # All (x, y) combos

    # === Step 4: Build 2D grid graph with real coordinates ===
    index_to_coord = {(i, j): (x, y) for i, x in enumerate(x_coords) for j, y in enumerate(y_coords)}
    G = nx.grid_2d_graph(len(x_coords), len(y_coords))
    G = nx.relabel_nodes(G, index_to_coord)

    # === Step 5: Mark detected dot nodes ===
    for point in G.nodes():
        G.nodes[point]['is_dot'] = point in dots_array
    
    return G, x_coords,y_coords

