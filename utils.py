# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:28:11 2023

@author: cruzguea
"""
import numpy as np
import vtk
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from os import path, getlogin, listdir
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import open3d as o3d
from datetime import datetime
from sklearn.model_selection import train_test_split
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import copy
from torch_geometric.data import Data, Database
from torch_geometric.transforms import SamplePoints, KNNGraph
import torch_geometric.transforms as T
import torch_geometric as tog
from DataLoaders import LoadedDataset, MyData
from pathlib import Path
import pandas as pd
import sys
import pdb
sys.path.append('D:\OneDrive - The University of Colorado Denver\R01_Grant\CranioRegistration\cranialphenotyping-LowerLandmarkTemplate\python')
from cpToolkit.cranioToolkit import CTImage

def FindFilesToProcess(inputFolder, filename):
    #make sure our input folder exists
    if not path.exists(inputFolder):
        raise ValueError('Input folder %s not found! Please check.'%inputFolder)

    #use the input folder and recursively search for any files that match our CT filename
    filesToProcess = [path for path in Path(inputFolder).rglob(filename)]

    #raise an error if we can't find any files
    if len(filesToProcess)==0:
        raise ValueError('No files in folder %s found!\nPlease check folder and input file name: %s'%(inputFolder,filename))
    else:
        print('Found %d files to process'%len(filesToProcess))
    
    return filesToProcess

def ReadPolyData(filename):
    if filename.endswith('.vtk'):
        reader = vtk.vtkPolyDataReader()
    else:
        reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def WritePolyData(data, filename):
    if filename.endswith('.vtk'):
        writer = vtk.vtkPolyDataWriter()
    else:
        writer = vtk.vtkXMLPolyDataWriter()
    # Saving landmarks
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(data)
    writer.Update()
    return

    
def AugmentTexture(data):
    texture = data * 255 #one too many / 255
    #add in gaussian noise
    p = 0.01
    noise = np.random.normal(loc = 0, scale = p * texture.mean(), size = texture.shape)
    #now let's randomly zero out some of the noise
    noise_percent = 0.5
    noise[np.random.choice(np.arange(texture.shape[0]), size = int(texture.shape[0] * noise_percent)),:] = 0
    texture = texture - noise

    return texture

def ComputeNormals(inputMesh, flip: int=0, split=False):
    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(inputMesh)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    filter.NonManifoldTraversalOff()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.SetFlipNormals(flip)
    filter.SetSplitting(split)
    filter.Update()
    outputMesh = filter.GetOutput()
    return outputMesh

def CleanPolyData(inputMesh, tol=0):
    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(inputMesh)
    filter.PointMergingOn()
    filter.ConvertLinesToPointsOff()
    filter.ConvertPolysToLinesOff()
    filter.SetTolerance(tol)
    filter.Update()
    outputMesh = filter.GetOutput()
    return outputMesh

def GetPointNormal(mesh, point):
    closestPoint = GetClosestVertex(mesh, point)
    normal = mesh.GetPointData().GetNormals().GetTuple(closestPoint)
    normalArray = np.array(normal)
    return normalArray

def FixDegerateRegion(inputRegion):
    ids = vtk.vtkIdList()
    ids.InsertNextId(0)

    extractFilter = vtk.vtkExtractCells()
    extractFilter.SetInputData(inputRegion)
    extractFilter.SetCellList(ids)
    extractFilter.Update()
    unstructuredGrid = extractFilter.GetOutput()

    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(unstructuredGrid)
    geometryFilter.Update()
    outputRegion = geometryFilter.GetOutput()
    return outputRegion

def GetMeshRegions(inputMesh):
    filter = vtk.vtkPolyDataConnectivityFilter()
    filter.SetInputData(inputMesh)
    filter.SetExtractionModeToSpecifiedRegions()

    regions = []
    regionId = 0

    while True:
        filter.AddSpecifiedRegion(regionId)
        filter.Update()

        region = vtk.vtkPolyData()

        region.DeepCopy(filter.GetOutput())
        # Make sure we got something
        if region.GetNumberOfCells() <= 0:
            break

        region = CleanPolyData(region)
        regions.append(region)
        filter.DeleteSpecifiedRegion(regionId)
        regionId += 1

    # sort regions by number of cells
    regions.sort(key=lambda x: x.GetNumberOfCells(), reverse=True)
    mainRegion = regions[0]
    smallRegions = regions[1:]

    return mainRegion, smallRegions

def GetBoundaryEdges(inputMesh):
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(inputMesh)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.Update()
    boundaryEdges = feature_edges.GetOutput()
    return boundaryEdges

def GetBoundaryPoint(inputMesh):
    boundaryEdges = GetBoundaryEdges(inputMesh)
    boundaryPoint = boundaryEdges.GetPoint(0)
    return boundaryPoint
def GetClosestVertex(data: vtk.vtkPolyData, point):
    locator = vtk.vtkPointLocator() # locate closest point on mesh
    locator.SetDataSet(data)
    locator.Update()
    locator.BuildLocator()
    return locator.FindClosestPoint(point)
def vtkPolyDataToNumpy(polydata,arrayName =  None, data_type = 'point'):
    if not arrayName:
        if data_type == 'point':
            numpyArray = vtk_to_numpy(polydata.GetPoints().GetData())
        elif data_type =='cell':
            raise ValueError('Cannot get generic cell data. Please specify an array.')
    else:
        if data_type == 'point':
            numpyArray = vtk_to_numpy(polydata.GetPointData().GetAbstractArray(arrayName))
        elif data_type =='cell':
            numpyArray = vtk_to_numpy(polydata.GetCellData().GetAbstractArray(arrayName))
    return numpyArray
def GetPointNormal(inputMesh, point):
    closestPoint = GetClosestVertex(inputMesh, point)
    normal = inputMesh.GetPointData().GetNormals().GetTuple(closestPoint)
    normalArray = np.array(normal)
    return normalArray

def GetAngleBetweenNormals(n1, n2):
    angleRadians = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))
    angleDegrees = np.degrees(angleRadians)
    return angleDegrees

def IsNormalDirectionInward(inputMesh):
    points = vtkPolyDataToNumpy(inputMesh, data_type='point')
    center = np.mean(points, axis=0)
    pointNormal = GetPointNormal(inputMesh, points[0])
    centerDirection = center - points[0]
    if np.dot(pointNormal, centerDirection) > 0:
        return True
    else:
        return False

def CorrectFlippedRegions(inputMesh):
    mainRegion, smallRegions = GetMeshRegions(inputMesh)
    mainRegion = ComputeNormals(mainRegion, flip=0)
    if IsNormalDirectionInward(mainRegion):
        mainRegion = ComputeNormals(mainRegion, flip=1)

    smallRegionsCorrected = []

    for smallRegion in smallRegions:
        if GetBoundaryEdges(smallRegion).GetNumberOfPoints() == 0:
            try:
                assert smallRegion.GetNumberOfCells() == 2, "Degenerate region contains more than 2 cells!"
                smallRegion = FixDegerateRegion(smallRegion)
            except AssertionError:
                print("Degenerate region contains more than 2 cells, skipping correction.")
                continue

        smallRegion = ComputeNormals(smallRegion, flip=0)
        boundaryPoint = GetBoundaryPoint(smallRegion)

        mainRegionNormal = GetPointNormal(mainRegion, boundaryPoint)
        smallRegionNormal = GetPointNormal(smallRegion, boundaryPoint)

        angle = GetAngleBetweenNormals(mainRegionNormal, smallRegionNormal)
        if angle > 90:
            smallRegion = ComputeNormals(smallRegion, flip=1)

        smallRegionsCorrected.append(smallRegion)

    appendFilter = vtk.vtkAppendPolyData()
    appendFilter.AddInputData(mainRegion)
    for region in smallRegionsCorrected:
        appendFilter.AddInputData(region)
    appendFilter.Update()
    outputMesh = appendFilter.GetOutput()
    outputMesh = CleanPolyData(outputMesh)

    return outputMesh
def closeMesh (mesh):

    # We eliminate duplicates, because they may be marked as boundaries incorrectly
    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()
    
    # We make sure that there are only triangle cells in the mesh
    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(mesh)
    filter.PassLinesOff()
    filter.PassVertsOff()
    filter.Update()
    mesh = filter.GetOutput()
    
    # Get edges
    filter = vtk.vtkFeatureEdges()
    filter.SetInputData(mesh)
    filter.ExtractAllEdgeTypesOff()
    filter.BoundaryEdgesOn()
    filter.Update()
    exteriorEdges = filter.GetOutput()
    
    # Triagulate edges
    filter = vtk.vtkDelaunay2D()
    filter.SetInputData(exteriorEdges)
    filter.SetProjectionPlaneMode(2) # VTK_BEST_FITTING_PLANE
    filter.Update()
    triangulatedEdges = filter.GetOutput()
    
    # Append meshes
    filter = vtk.vtkAppendPolyData()
    filter.AddInputData(mesh)
    filter.AddInputData(triangulatedEdges)
    filter.Update()
    closedMesh = filter.GetOutput()
    
    # Fill any small holes that may remain
    filter = vtk.vtkFillHolesFilter()
    filter.SetInputData(closedMesh)
    filter.SetHoleSize(1e6)
    filter.Update()
    closedMesh = filter.GetOutput()
    
    # Clean mesh
    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(closedMesh)
    filter.Update()
    closedMesh = filter.GetOutput()
    
    # We make sure that there are only triangle cells in the mesh
    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(closedMesh)
    filter.PassLinesOff()
    filter.PassVertsOff()
    filter.Update()
    closedMesh = filter.GetOutput()

    smoothFilter = vtk.vtkWindowedSincPolyDataFilter()
    smoothFilter.SetInputData(closedMesh)
    smoothFilter.SetNumberOfIterations(50)
    # smoothFilter.SetRelaxationFactor(0.01)
    smoothFilter.FeatureEdgeSmoothingOn()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()
    outputSurface = smoothFilter.GetOutput()

    # Update normals
    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(outputSurface)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    filter.NonManifoldTraversalOn()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.Update()
    closedMesh = filter.GetOutput()
    
    return closedMesh


# 3. Recuperar datos de VTK
def from_vtk(polydata):
    positions = vtk_to_numpy(polydata.GetPoints().GetData())
    rgb = vtk_to_numpy(polydata.GetPointData().GetScalars())
    normals = vtk_to_numpy(polydata.GetPointData().GetNormals())
    return torch.tensor(positions), torch.tensor(rgb), torch.tensor(normals)

def CellToPointData(mesh):
    ptc = vtk.vtkCellDataToPointData()
    ptc.SetInputData(mesh)
    ptc.Update()
    return ptc.GetOutput()

def convert_to_graph(image_vtp, name=None, use_texture=False):
    '''
        Function to convert a 3D photograph (VTP mesh) and its landmarks to a graph
        According to https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
        x = node features
        y = labels
        pos = node positions
        edge_indices = COO format of graph 
    '''
    pos = vtk_to_numpy(image_vtp.GetPoints().GetData())
    # pos = pos - np.mean(pos,axis=0)
    # transform = sitk.Similarity3DTransform()
    # transform.SetMatrix(R)
    # transform.SetTranslation(trns)
    # pos2 = np.matmul((pos-np.mean(pos,axis=0)),np.array(R).reshape(3,3).transpose(1,0))+trns+np.mean(pos,axis=0)
    # pos2 = np.matmul((pos),np.array(R).reshape(3,3).transpose(1,0))+trns
    x = vtk_to_numpy(image_vtp.GetPointData().GetArray('Normals'))
    edge_table = get_edges_of_mesh(image_vtp)
    edge_indices = convert_to_coo(edge_table)
    # node_weights = calc_node_weights(torch.tensor(pos))
    x = torch.tensor(x)
    
    edge_indices, _ = tog.utils.remove_self_loops(edge_indices)
    row, col = edge_indices
    edge_weights = np.linalg.norm(pos[row] - pos[col], axis=1)
    
    data = MyData(x = x, pos = torch.tensor(pos), edge_index = torch.tensor(edge_indices, dtype = torch.long), num_nodes = len(pos), edge_weight =torch.tensor(edge_weights), imageID=name)

    if use_texture:
            #normalize the texture
            texture = vtk_to_numpy(image_vtp.GetPointData().GetArray('Texture'))
            texture[np.isnan(texture)]=0
            data.x  = torch.cat((torch.tensor(x), torch.tensor(texture)/255), dim = 1)
    #now adjust the batch
    data.batch = torch.zeros(data.pos.shape[0], dtype = torch.int64)
    # dataR.batch = torch.zeros(data.pos.shape[0], dtype = torch.int64)
    return data#, dataR

def FindClosestPoint(data, point):
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(data)
    locator.Update()
    return locator.FindClosestPoint((point[0], point[1], point[2]))

def InterpolateTextureToPoints(mesh):
    textures = np.zeros([mesh.GetNumberOfPoints(), 3])
    celltextures = mesh.GetCellData().GetArray('Texture')
    for point in range(mesh.GetNumberOfPoints()):
        #for each cell
        cellidlist = vtk.vtkIdList()
        mesh.GetPointCells(point, cellidlist)
        textures[point, :] = np.mean(np.array([celltextures.GetTuple(cellidlist.GetId(cellid)) for cellid in range(cellidlist.GetNumberOfIds())]), axis = 0)/255
    
    if True in np.isnan(textures):
        position = np.where(np.isnan(textures))[0]
        position1 = (position.reshape(int(len(position)/3),3))[:,0]
        for i in position1:
            textures[i,:]= (textures[i-1,:]+textures[i-2,:])/2

    textureArray = vtk.vtkFloatArray()
    textureArray.SetName('Texture')
    textureArray.SetNumberOfComponents(3)
    for point in range(mesh.GetNumberOfPoints()):
        point_id = FindClosestPoint(mesh, mesh.GetPoint(point))
        if True in np.isnan(textures[point_id,:]):
            textures[point_id,:]=textures[point_id-1,:]
        textureArray.InsertNextTuple3(textures[point_id,0],textures[point_id,1],textures[point_id,2])
    
    mesh.GetPointData().AddArray(textureArray)
    return mesh

def calc_node_weights(pos):
    return torch.empty(pos.shape[0])
    # pdb.set_trace()
    # dist = torch.cdist(pos, pos)
    # average_dist = torch.sum(dist, dim = 0)/(dist.shape[0]-1)
    # normalized_dist = (average_dist - average_dist.min())/ (average_dist.max() - average_dist.min())
    # return torch.stack([normalized_dist,1-normalized_dist], dim = 1)

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

def ApplyTransform(data, transform):
    # Creating a copy of the input meshes
    a = vtk.vtkPolyData()
    a.DeepCopy(data)
    data = a

    for p in range(data.GetNumberOfPoints()):
        coords = np.array(data.GetPoint(p))
        newCoords = transform.TransformPoint(coords.astype(np.float64))
        data.GetPoints().SetPoint(p, newCoords[0], newCoords[1], newCoords[2])
    # Recalculating the normals and saving

    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(data)
    filter.Update()
    data = filter.GetOutput()

    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(data)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    # filter.NonManifoldTraversalOn()
    filter.NonManifoldTraversalOff()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.SplittingOff()
    filter.Update()
    data = filter.GetOutput()
    return data

def ApplyTransformLand(data, transform):
    # Creating a copy of the input meshes
    a = vtk.vtkPolyData()
    a.DeepCopy(data)
    data = a

    for p in range(data.GetNumberOfPoints()):
        coords = np.array(data.GetPoint(p))
        newCoords = transform.TransformPoint(coords.astype(np.float64))
        data.GetPoints().SetPoint(p, newCoords[0], newCoords[1], newCoords[2])
    # Recalculating the normals and saving

    return data

def ApplyTransformPC(pc, rttn, trns, scl=None):

    outPC = np.zeros(pc.shape)
    transform = sitk.Similarity3DTransform()
    center = np.mean(pc, axis=0).astype(np.float64)

    # pc = pc- center
    # if scl:
    #     transform.SetScale(scl)
    #     R *= s
    transform.SetMatrix(rttn)
    # transform.SetRotation(rttn[n])
    transform.SetCenter(center)
    transform.SetTranslation(trns)
    for p in range(len(pc)):
        newCoords = transform.TransformPoint(pc[p].astype(np.float64))
        outPC[p,:] = newCoords

    return outPC

def DownsampleMesh(mesh, target_reduction = 0.1, angle=30, use_texture = True):

    #move the texture to the points!
    mesh = CellToPointData(mesh)

    filter = vtk.vtkDecimatePro()
    filter.SetInputData(mesh)
    filter.SetTargetReduction(1-target_reduction)
    filter.SetSplitAngle(angle)
    filter.Update()
    decimated_mesh = filter.GetOutput()

    # Calculating normals
    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(decimated_mesh)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    # filter.NonManifoldTraversalOn()
    filter.NonManifoldTraversalOff()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.SplittingOff()
    filter.Update()
    decimated_mesh = filter.GetOutput()

    return decimated_mesh

def sitkToVtkImage(sitkImage):

    numpyImage = sitk.GetArrayViewFromImage(sitkImage)
    dataArray = numpy_support.numpy_to_vtk(num_array=numpyImage.ravel(), deep=True)
    
    vtkImage = vtk.vtkImageData()
    vtkImage.SetSpacing(sitkImage.GetSpacing()[0], sitkImage.GetSpacing()[1], sitkImage.GetSpacing()[2])
    vtkImage.SetOrigin(sitkImage.GetOrigin()[0], sitkImage.GetOrigin()[1], sitkImage.GetOrigin()[2])
    vtkImage.SetExtent(0, numpyImage.shape[2]-1, 0, numpyImage.shape[1]-1, 0, numpyImage.shape[0]-1)
    vtkImage.GetPointData().SetScalars(dataArray)
    
    return vtkImage
    
def vtkToSitkImage(vtkImage):

    numpyImage = numpy_support.vtk_to_numpy(vtkImage.GetPointData().GetScalars()).reshape(vtkImage.GetDimensions()[::-1])
    
    sitkImage = sitk.GetImageFromArray(numpyImage)
    sitkImage.SetOrigin(vtkImage.GetOrigin())
    sitkImage.SetSpacing(vtkImage.GetSpacing())
    
    return sitkImage

def meshToVolume (mesh):
    
    # Fills all holes in mesh
    mesh = closeMesh(mesh)  
    
    # Gets bounds, spacing, origin, and dimensions of mesh
    bounds = np.array(mesh.GetBounds())
    #give the bounds a bit of extra space
    bounds = bounds * 1.1
    spacing = (1, 1, 1)
    origin = bounds[0::2]
    dims = (bounds[1::2] - origin) / spacing
    dims = dims.astype(np.int32)
    
    # Creates image with above defined spacing, origin, dimensions, and extent; allocates scalars
    image = vtk.vtkImageData()
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDimensions(dims)
    image.SetExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    image.AllocateScalars(3, 1) # vtk.VTK_UNSIGNED_CHAR
    
    # Fills the image with white voxels (outValue is used for background)
    inValue = 1.0
    outValue = 0.0
    
    for i in range(image.GetNumberOfPoints()):
        image.GetPointData().GetScalars().SetTuple1(i, inValue)
    
    # Converts PolyData (mesh) to ImageStencil, then converts ImageStencil to ImageData
    polyStencil = vtk.vtkPolyDataToImageStencil()
    imageStencil = vtk.vtkImageStencil()
    
    polyStencil.SetInputData(mesh)
    polyStencil.SetOutputOrigin(image.GetOrigin())
    polyStencil.SetOutputSpacing(image.GetSpacing())
    polyStencil.SetOutputWholeExtent(image.GetExtent())
    polyStencil.Update()
    
    imageStencil.SetInputData(image)
    imageStencil.SetStencilData(polyStencil.GetOutput())
    imageStencil.ReverseStencilOff()
    imageStencil.SetBackgroundValue(outValue)
    imageStencil.Update()
       
    # Returns ImageData output
    image = imageStencil.GetOutput()
    return image

def closeMesh (mesh):

    # We eliminate duplicates, because they may be marked as boundaries incorrectly
    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()
    
    # We make sure that there are only triangle cells in the mesh
    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(mesh)
    filter.PassLinesOff()
    filter.PassVertsOff()
    filter.Update()
    mesh = filter.GetOutput()
    
    # Get edges
    filter = vtk.vtkFeatureEdges()
    filter.SetInputData(mesh)
    filter.ExtractAllEdgeTypesOff()
    filter.BoundaryEdgesOn()
    filter.Update()
    exteriorEdges = filter.GetOutput()
    
    # Triagulate edges
    filter = vtk.vtkDelaunay2D()
    filter.SetInputData(exteriorEdges)
    filter.SetProjectionPlaneMode(2) # VTK_BEST_FITTING_PLANE
    filter.Update()
    triangulatedEdges = filter.GetOutput()
    
    # Append meshes
    filter = vtk.vtkAppendPolyData()
    filter.AddInputData(mesh)
    filter.AddInputData(triangulatedEdges)
    filter.Update()
    closedMesh = filter.GetOutput()
    
    # Fill any small holes that may remain
    filter = vtk.vtkFillHolesFilter()
    filter.SetInputData(closedMesh)
    filter.SetHoleSize(1e6)
    filter.Update()
    closedMesh = filter.GetOutput()
    
    # Clean mesh
    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(closedMesh)
    filter.Update()
    closedMesh = filter.GetOutput()
    
    # We make sure that there are only triangle cells in the mesh
    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(closedMesh)
    filter.PassLinesOff()
    filter.PassVertsOff()
    filter.Update()
    closedMesh = filter.GetOutput()

    smoothFilter = vtk.vtkWindowedSincPolyDataFilter()
    smoothFilter.SetInputData(closedMesh)
    smoothFilter.SetNumberOfIterations(50)
    # smoothFilter.SetRelaxationFactor(0.01)
    smoothFilter.FeatureEdgeSmoothingOn()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()
    outputSurface = smoothFilter.GetOutput()


    # Update normals
    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(outputSurface)
    filter.ComputeCellNormalsOn()
    filter.ComputePointNormalsOn()
    filter.NonManifoldTraversalOn()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.Update()
    closedMesh = filter.GetOutput()
    
    return closedMesh

def CreateMeshFromBinaryImage(binaryImage, insidePixelValue=1):
    """
    Uses the marching cubes algorithm to create a surface model from a binary image

    Parameters
    ----------
    binaryImage: sitkImage
        The binary image
    insidePixelValue: {int, float}
        The pixel value to use for mesh creation

    Returns
    -------
    vtkPolyData
        The resulting surface model
    """

    numpyImage = sitk.GetArrayViewFromImage(binaryImage).astype(np.ubyte)
    
    dataArray = numpy_support.numpy_to_vtk(num_array=numpyImage.ravel(),  deep=True,array_type=vtk.VTK_UNSIGNED_CHAR)

    vtkImage = vtk.vtkImageData()
    vtkImage.SetSpacing(binaryImage.GetSpacing()[0], binaryImage.GetSpacing()[1], binaryImage.GetSpacing()[2])
    vtkImage.SetOrigin(binaryImage.GetOrigin()[0], binaryImage.GetOrigin()[1], binaryImage.GetOrigin()[2])
    vtkImage.SetExtent(0, numpyImage.shape[2]-1, 0, numpyImage.shape[1]-1, 0, numpyImage.shape[0]-1)
    vtkImage.GetPointData().SetScalars(dataArray)

    filter = vtk.vtkMarchingCubes()
    filter.SetInputData(vtkImage)
    filter.SetValue(0, insidePixelValue)
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkGeometryFilter()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    return mesh

def MeshReg(source, target):
    # ICP for alignment
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(100)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()
    return icp

def AppliedMeshTrans(Mesh, transform):
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(Mesh)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    return transform_filter.GetOutput()

def GetLandmarkNames(landmarks):
    arr = landmarks.GetPointData().GetAbstractArray('LandmarkName')
    return np.array([arr.GetValue(x) for x in range(arr.GetNumberOfValues())])

def SelectLandmarks(landmarks, selection):
    try:
        landmark_names = GetLandmarkNames(landmarks)
    except:
        pdb.set_trace()
    if not np.all(np.isin(selection, landmark_names)):
        raise ValueError('Incorrect landmark selection! Please check input.')

    ids = [int(np.where(landmark_names == x)[0]) for x in selection]
    out_landmarks = vtk.vtkPolyData()
    out_landmarks.SetPoints(vtk.vtkPoints())
    for id in ids:
        out_landmarks.GetPoints().InsertNextPoint(landmarks.GetPoints().GetPoint(id))
    return out_landmarks

def GenerateMesh(data):
    data.edge_index = tog.utils.remove_self_loops(data.edge_index)[0]
    pos= data.pos
    edges = data.edge_index
    normals =  data.x[:,:3]
    textures = data.x[:,3:]
    edge_dict = {k: [] for k in range(data.num_nodes)}
    #build the whole edge list as a dict!
    for k,v in zip(edges[0],edges[1]):
        edge_dict[k.item()].append(v.item())
    # now sort through and generate the faces
    faces = []
    for k,v in edge_dict.items():
        for j in v:
            [faces.append((k, j, x)) for x in edge_dict[j] if k in edge_dict[x]]

    #now construct the cells and the points
    cellArray = vtk.vtkCellArray()
    for face in faces:
        cellArray.InsertNextCell(3)
        cellArray.InsertCellPoint(face[0])
        cellArray.InsertCellPoint(face[1])
        cellArray.InsertCellPoint(face[2])

    textureArray = vtk.vtkFloatArray()
    textureArray.SetName('Texture')
    textureArray.SetNumberOfComponents(3)
    for i in range(len(textures)):
        textureArray.InsertNextTuple3(textures[i,0],textures[i,1],textures[i,2])


    points = vtk.vtkPoints()
    for point in pos:
        points.InsertNextPoint(point[0], point[1], point[2])
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetPolys(cellArray)
    polyData.GetPointData().AddArray(textureArray)
    
    
    norms= vtk.vtkFloatArray()
    norms.SetName('Normals')
    norms.SetNumberOfComponents(3)
    for i in range(normals.shape[0]):
        norms.InsertNextTuple3(normals[i,0], normals[i,1], normals[i,2])
    polyData.GetPointData().AddArray(norms)

    return polyData

if __name__ == "__main__":
    transform_name = 'final_transform.tfm'
    mesh_name = 'landmarks.vtp'
    mesh_name2 = 'photo.vtp'
    landmark_name = 'CranialBaseLandmarks.vtp'
    # mesh_name = 'ExternalHeadSurface-updated.vtp'
    tempDir = r'D:\OneDrive - The University of Colorado Denver\R01_Grant\CranioRegistration\cranialphenotyping-LowerLandmarkTemplate\python\CT_template'
    basedirs = [r'D:\OneDrive - The University of Colorado Denver\3DMD_Processed_Image_Database_REDO - EC']
    outdir = r'D:\OneDrive - The University of Colorado Denver\DataMeshCHCO_Full'
    # skiplist = pd.read_csv(r'D:\OneDrive - The University of Colorado Denver\CT_Data_CHCO\Skiplist-Updated-9-12.csv')
    
    files_to_process= []
    for basedir in basedirs:
        files_to_process += FindFilesToProcess(basedir,mesh_name)
    # files_to_process = [x for x in files_to_process if x.parent.name not in skiplist['Skiplist'].values]
    files_to_process = [x for x in files_to_process if not 'X410-150128105746' == x.parent.name]

    datadir = r'D:\OneDrive - The University of Colorado Denver\DataMeshCHCO4_25'
    files1 = FindFilesToProcess(datadir,'*_R_meshHeadTexLand.pt')
    filesC = []
    for i, file1 in enumerate(files1):
        for j, file2 in enumerate(files_to_process):
            if file1.name[:-21]==file2.parent.name:
                filesC.append(file2)
    rttnM = []
    trnsV = []
    sclrF = []
    normVec = []
    DatabaseReg = []
    DatabaseTra = []
    DatabaseRegWS = []
    DatabaseTraWS = []
    DatabaseRegPC = []
    DatabaseTraPC = []
    dataAu_size = 1
    isGraphDB= True
    GraphTransPC=True
    PointsNumber=500
    MassCenter = np.array([9.031361,-177.76645,-99.50831])
    final_selection = ['NASION','TRAGION_RIGHT','TRAGION_LEFT', 'OPISTHOCRANION']
    edges=[]
    points =[]
    cells = []
    # landRef = ReadPolyData(path.join(tempDir, landmark_name))
    for i, file in enumerate(filesC):
        try:
            print(f'On {i} out of {len(filesC)}: {file.parent.name}')
            mesh1 = ReadPolyData(str(file.with_name(mesh_name2)))
            # edges.append(mesh1.GetNumberOfCells())
            # points.append(mesh1.GetNumberOfPoints())

            if i ==0:
                landRef = ReadPolyData(str(file.with_name(mesh_name)))
                transform = sitk.Euler3DTransform()
                transform.SetTranslation(-np.mean(vtk_to_numpy(SelectLandmarks(landRef, final_selection).GetPoints().GetData()), axis=0).astype(np.float64))
                landRef = ApplyTransformLand(landRef, transform)
                landmarks = vtk.vtkPolyData()
                landmarks.DeepCopy(landRef)
                meshRef = vtk.vtkPolyData()
                meshRef.DeepCopy(mesh1)
                meshRef = ApplyTransform(meshRef, transform)
                mesh_moved = ApplyTransform(mesh1, transform)
            else:
                landmarks = ReadPolyData(str(file.with_name(mesh_name)))
                transform = MeshReg(mesh1, meshRef)
                matrix = transform.GetMatrix()  # vtkMatrix4x4 object
                # Convert to numpy array
                mat_np = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
                rotation = mat_np[:3, :3]
                # The translation vector is the last column of the first 3 rows
                translation = mat_np[:3, 3]
                transform2 = sitk.Euler3DTransform()
                transform2.SetTranslation(translation.astype(np.float64))
                transform2.SetMatrix(rotation.flatten().astype(np.float64))
                landmarks = ApplyTransformLand(landmarks, transform2)
                mesh_moved = AppliedMeshTrans(mesh1, transform)


            mesh_moved2=DownsampleMesh(mesh_moved, target_reduction=0.25)
            mesh = CellToPointData(mesh_moved2)
            
            mesh = CleanPolyData(mesh, tol=0.001)
            mesh = ComputeNormals(mesh, split=True) 
            preprocessed_mesh = CorrectFlippedRegions(mesh)
            
            extractEdges = vtk.vtkExtractEdges()
            extractEdges.SetInputData(mesh)
            extractEdges.Update()

            edgesPolyData = extractEdges.GetOutput()
            numEdges = edgesPolyData.GetNumberOfLines()
            
            edges.append(numEdges)
            cells.append(mesh.GetNumberOfCells())
            points.append(mesh.GetNumberOfPoints())
            graphReg = convert_to_graph(preprocessed_mesh, file.parent.name, use_texture=True)
            # graphReg.landmarks=vtk_to_numpy(landmarks.GetPoints().GetData())
            
            # torch.save(graphReg, path.join(outdir,file.parent.name+'_R_'+'meshHeadTexLand.pt'))
            # torch.save(graphTra, path.join(outdir,file.parent.name+str(n)+'_T_'+'meshHeadTexLand.pt'))
        except ValueError:
            print("Oops!  That was no valid number.")
    print("Average of number of points: ", np.mean(points)," std: ", np.std(points), 'min:', np.min(points), 'max:', np.max(points))
    print("Average of edges: ", np.mean(edges)," std: ", np.std(edges), 'min:', np.min(edges), 'max:', np.max(edges))
    print("Average of cells: ", np.mean(cells)," std: ", np.std(cells), 'min:', np.min(cells), 'max:', np.max(cells))
