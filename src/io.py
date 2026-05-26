import vtk
from pathlib import Path
from os import path
import pandas as pd

def FindFilesToProcess(inputFolder, verbose=False):
    # validar carpeta
    if not path.exists(inputFolder):
        raise ValueError(f'Input folder {inputFolder} not found! Please check.')

    # buscar todos los .vtp recursivamente
    all_files = list(Path(inputFolder).rglob("*.vtp"))

    if len(all_files) == 0:
        raise ValueError(f'No .vtp files found in {inputFolder}')

    # agrupar por carpeta
    folder_dict = {}

    for file_path in all_files:
        folder = file_path.parent

        if folder not in folder_dict:
            folder_dict[folder] = []

        folder_dict[folder].append(file_path.name)

    # construir tabla
    rows = []

    for folder, files in folder_dict.items():
        files_set = set(files)
        if folder.parts[-1] != inputFolder:
            parentsName=folder.parts[-2]
            row = {
                "parent_names": parentsName,
                "folder_names": folder.parts[-1],
                "has_landmarks": int("landmarks.vtp" in files_set),
                "has_photo": int("photo.vtp" in files_set),
                "has_photo_raw": int("photo-raw.vtp" in files_set),
                "folder_path": str(folder),
            }
        else:
            row = {
                "parent_names": folder.parts[-1],  # nombres de carpetas padre
                "has_landmarks": int("landmarks.vtp" in files_set),
                "has_photo": int("photo.vtp" in files_set),
                "has_photo_raw": int("photo-raw.vtp" in files_set),
                "folder_path": str(folder),
            }

        rows.append(row)

    df = pd.DataFrame(rows)
    if verbose:
        print(f'Found {len(all_files)} .vtp files')
        print(f'Total folders detected: {len(df)}')
        print(f'Total landmarks files detected: {df['has_landmarks'].sum()}')
        print(f'Total photo files detected: {df['has_photo'].sum()}')
        print(f'Total photo-raw files detected: {df['has_photo_raw'].sum()}')
        print(f'Folders with preprocessed photos: {(df['has_landmarks'] & df['has_photo']).sum()}')
        print(f'Folders with all the files: {(df['has_landmarks'] & df['has_photo'] & df['has_photo_raw']).sum()}')

    return df

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
