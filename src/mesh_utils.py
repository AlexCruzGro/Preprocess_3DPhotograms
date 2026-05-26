import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import torch
import SimpleITK as sitk
import pdb
from collections import defaultdict
from itertools import combinations
from src.graph_utils import MyData

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

    return data

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
    dataArray = numpy_to_vtk(num_array=numpyImage.ravel(), deep=True)
    
    vtkImage = vtk.vtkImageData()
    vtkImage.SetSpacing(sitkImage.GetSpacing()[0], sitkImage.GetSpacing()[1], sitkImage.GetSpacing()[2])
    vtkImage.SetOrigin(sitkImage.GetOrigin()[0], sitkImage.GetOrigin()[1], sitkImage.GetOrigin()[2])
    vtkImage.SetExtent(0, numpyImage.shape[2]-1, 0, numpyImage.shape[1]-1, 0, numpyImage.shape[0]-1)
    vtkImage.GetPointData().SetScalars(dataArray)
    
    return vtkImage
    
def vtkToSitkImage(vtkImage):

    numpyImage = vtk_to_numpy(vtkImage.GetPointData().GetScalars()).reshape(vtkImage.GetDimensions()[::-1])
    
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
    
    dataArray = numpy_to_vtk(num_array=numpyImage.ravel(),  deep=True,array_type=vtk.VTK_UNSIGNED_CHAR)

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
    # icp.SetMaximumNumberOfLandmarks(5000)
    icp.SetMeanDistanceModeToRMS()
    # icp.StartByMatchingCentroidsOn()
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




def GenerateMesh(graph: MyData) -> vtk.vtkPolyData:
    
    if hasattr(graph, 'pos') and hasattr(graph, 'x') and graph.x.size(1) > 3:
        points = numpy_to_vtk(graph.pos.clone().cpu().numpy())
        if graph.x.shape[1] == 6:
            normals = numpy_to_vtk(graph.x[:, :3].clone().cpu().numpy())
            normals.SetName('Normals')
            texture = numpy_to_vtk(graph.x[:, 3:].clone().cpu().numpy())
            texture.SetName('Texture')

        edgeList = graph.edge_index.clone().cpu().t().tolist()
        faces = set()
        adj = defaultdict(set)

        for u, v in edgeList:
            adj[u].add(v)

        for u in adj:
            if len(adj[u]) < 2:
                continue
            for v, w in combinations(adj[u], 2):
                if w in adj[v]:
                    face = tuple(sorted((u, v, w)))
                    faces.add(face)

        mesh = vtk.vtkPolyData()

        mesh.SetPoints(vtk.vtkPoints())
        mesh.GetPoints().SetData(points)
        if graph.x.shape[1] == 6:
            mesh.GetPointData().AddArray(texture)
            mesh.GetPointData().AddArray(normals)

        mesh.SetPolys(vtk.vtkCellArray())
        for face in faces:
            mesh.GetPolys().InsertNextCell(3, face)
    else:
        points = numpy_to_vtk(graph.x.clone().cpu().numpy())
        
        edgeList = graph.edge_index.clone().cpu().t().tolist()
        faces = set()
        adj = defaultdict(set)

        for u, v in edgeList:
            adj[u].add(v)

        for u in adj:
            if len(adj[u]) < 2:
                continue
            for v, w in combinations(adj[u], 2):
                if w in adj[v]:
                    face = tuple(sorted((u, v, w)))
                    faces.add(face)

        mesh = vtk.vtkPolyData()

        mesh.SetPoints(vtk.vtkPoints())
        mesh.GetPoints().SetData(points)
        if graph.x.shape[1] == 6:
            mesh.GetPointData().AddArray(texture)
            mesh.GetPointData().AddArray(normals)

        mesh.SetPolys(vtk.vtkCellArray())
        for face in faces:
            mesh.GetPolys().InsertNextCell(3, face)

    return mesh