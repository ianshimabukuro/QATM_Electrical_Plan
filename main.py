from utils import *
from graph_functions import *

template_dir = 'template/'
image_path = 'sample/electrical_plan.png'
dataset = ImageDataset(Path(template_dir), image_path, thresh_csv='thresh_template.csv')
model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=25, use_cuda=False)
scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
boxes, indices, dots, keep = nms_multi(scores, w_array, h_array, thresh_list)
d_img = plot_result_multi(dataset.image_raw, boxes, indices, show=True, save_name='result_sample.png')

scale = 10.3 #dpi
G, x_coords, y_coords = dots_to_hanan(dots,keep, scale)

# === Step 6: Set up for drawing ===

pos = {node: node for node in G.nodes()}  # (x, y) layout
node_colors = [
    'red' if G.nodes[n].get('is_dot') else 'lightgray'
    for n in G.nodes()
]



path = nx.shortest_path(G, source=(5417.8, 5108.8), target=(18437.0, 18488.5))


# === Step 7: Dynamic figure size ===
x_range = max(x_coords) - min(x_coords) if x_coords else 1
y_range = max(y_coords) - min(y_coords) if y_coords else 1
scale = 0.05
figsize = (max(4, x_range * scale), max(4, y_range * scale))  # minimum figure size

plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=5)
nx.draw_networkx_edges(G, pos, edge_color='black', width=0.1)
edge_path = list(zip(path[:-1], path[1:]))  # edges from node pairs
nx.draw_networkx_edges(G, pos, edgelist=edge_path, edge_color='red', width=2)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.tight_layout()
plt.title("Hanan Grid: Red = Detections, Gray = Grid", fontsize=12)
plt.show()
