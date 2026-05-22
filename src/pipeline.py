from src.io import read_polydata, write_polydata
from src.mesh_utils import compute_normals, clean_mesh
from src.graph_utils import mesh_to_graph

def process_subject(input_file, output_file, config):
    mesh = read_polydata(input_file)

    if config["clean"]:
        mesh = clean_mesh(mesh)

    if config["normals"]:
        mesh = compute_normals(mesh)

    graph = mesh_to_graph(mesh)

    write_polydata(mesh, output_file)

    return graph
