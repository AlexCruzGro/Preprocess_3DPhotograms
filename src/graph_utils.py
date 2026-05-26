import torch
from vtk.util.numpy_support import vtk_to_numpy
from torch_geometric.data import Data
import torch_geometric as tog
import numpy as np
import vtk

class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        #along y we want a new dimension to handle the spherical maps
        if key == 'y':
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
    def __len__(self):
        return 1

def calc_node_weights(pos):
    return torch.empty(pos.shape[0])

def get_edges_of_mesh( mesh):
    
    '''
    Construct an edge list using COO format
    '''
    edge_table = {}
    for point in range(mesh.GetNumberOfPoints()):
        # print(f'Extracting edges for point {point} out of {mesh.GetNumberOfPoints()}', end = '\r')
        #for each cell
        cellidlist = vtk.vtkIdList()
        mesh.GetPointCells(point, cellidlist)
        points = []
        for cellid in range(cellidlist.GetNumberOfIds()):
            #find the points for each cell
            pointidlist = vtk.vtkIdList()
            mesh.GetCellPoints(cellidlist.GetId(cellid), pointidlist) # get cell ids
            #get the actual points belonging to the cells
            points += [pointidlist.GetId(x) for x in range(pointidlist.GetNumberOfIds())]
        #only take each point once
        edge_table[point] = list(np.unique(points))
    return edge_table

def convert_to_coo(edge_table):
    # print('Converting format to coo...')
    in_edges = []
    out_edges = []
    for key, val in edge_table.items():
        in_edges += [key] * len(val)
        out_edges += val
    return np.array([in_edges, out_edges])

def convert_to_graph(image_vtp, name=None, use_texture=False, landmarks=[]):
    '''
        Function to convert a 3D photograph (VTP mesh) and its landmarks to a graph
        According to https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
        x = node features
        y = labels
        pos = node positions
        edge_indices = COO format of graph 
    '''
    pos = vtk_to_numpy(image_vtp.GetPoints().GetData())
    normals = vtk_to_numpy(image_vtp.GetPointData().GetArray('Normals'))
    edge_table = get_edges_of_mesh(image_vtp)
    edge_indices = convert_to_coo(edge_table)
    # node_weights = calc_node_weights(torch.tensor(pos))
    normals = torch.tensor(normals)
    
    if landmarks:
        pointsLand = torch.tensor(vtk_to_numpy(landmarks.GetPoints().GetData()))
    else:
        pointsLand = torch.empty(0)
    
    edge_indices, _ = tog.utils.remove_self_loops(edge_indices)
    row, col = edge_indices
    edge_weights = np.linalg.norm(pos[row] - pos[col], axis=1)
    
    data = MyData(x = normals, pos = torch.tensor(pos), edge_index = torch.tensor(edge_indices, dtype = torch.long), num_nodes = len(pos), edge_weight =torch.tensor(edge_weights), imageID=name, landmarks=pointsLand)

    if use_texture:
            #normalize the texture
            texture = vtk_to_numpy(image_vtp.GetPointData().GetArray('Texture'))
            texture[np.isnan(texture)]=0
            data.texture = torch.tensor(texture)/255
            #torch.cat((torch.tensor(x), torch.tensor(texture)/255), dim = 1)
    #now adjust the batch
    data.batch = torch.zeros(data.pos.shape[0], dtype = torch.int64)
    return data