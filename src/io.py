import vtk
from pathlib import Path
from os import path

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
