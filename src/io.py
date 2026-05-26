import vtk
from pathlib import Path
from os import path, walk
import pandas as pd

def FindFilesToProcess(inputFolder, verbose=False):
    if not path.exists(inputFolder):
        raise ValueError(f'Input folder {inputFolder} not found! Please check.')

    target_files = {"landmarks.vtp", "photo.vtp", "photo-raw.vtp"}
    folder_dict = {}
    total_files = 0

    for folder, _, filenames in walk(inputFolder):
        matches = target_files.intersection(filenames)
        if matches:
            folder_dict[Path(folder)] = matches
            total_files += len(matches)

    if total_files == 0:
        raise ValueError(f'No target .vtp files found in {inputFolder}')

    rows = []

    for folder, files in folder_dict.items():
        if folder.parts[-1] != inputFolder:
            parentsName=folder.parts[-2]
            row = {
                "parent_names": parentsName,
                "folder_names": folder.parts[-1],
                "has_landmarks": int("landmarks.vtp" in files),
                "has_photo": int("photo.vtp" in files),
                "has_photo_raw": int("photo-raw.vtp" in files),
                "folder_path": str(folder),
            }
        else:
            row = {
                "parent_names": folder.parts[-1],  # nombres de carpetas padre
                "has_landmarks": int("landmarks.vtp" in files),
                "has_photo": int("photo.vtp" in files),
                "has_photo_raw": int("photo-raw.vtp" in files),
                "folder_path": str(folder),
            }

        rows.append(row)

    df = pd.DataFrame(rows)
    if verbose:
        print(f'Found {total_files} .vtp files')
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
