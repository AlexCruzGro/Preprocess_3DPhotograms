from src.io import ReadPolyData, WritePolyData
import os
import random
import numpy as np
import torch
from pathlib import Path
from src.mesh_utils import *
from src.graph_utils import convert_to_graph
import gc


def freeze_seeds(seed: int = 42):
    """Freeze random seeds to improve reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Some ops have multiple deterministic implementations depending on version.
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def preprocess_mesh(mesh, config):
    
    mesh = CellToPointData(mesh)

    if config['clean']:
        mesh = CleanPolyData(mesh, tol=config['tolerance'])
    if config['normals']:
        mesh = ComputeNormals(mesh, split=True) 
    if config['fix_normals']:
        mesh = CorrectFlippedRegions(mesh)
    
    return mesh

def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def files_processed(output_dir, config, vtp_filename, save_landmarks=False):
    expected_files = ()

    if config["save_graphs"]:
        expected_files += (os.path.join(output_dir, "graph.pt"),)
    if config["save_features"]:
        expected_files += (
            os.path.join(output_dir, "normals.pt"),
            os.path.join(output_dir, "coordinates.pt"),
            os.path.join(output_dir, "edge_weights.pt"),
            os.path.join(output_dir, "textures.pt"),
        )
        if save_landmarks:
            expected_files += (os.path.join(output_dir, "landmarks.pt"),)
    if config["save_vtp_files"]:
        expected_files += (os.path.join(output_dir, vtp_filename),)
        if save_landmarks:
            expected_files += (os.path.join(output_dir, "landmarks.vtp"),)

    return bool(expected_files) and all(os.path.exists(file) for file in expected_files)

def preprocess_pipeline(input_file, output_file, config, photo=False, landmark=False, photoraw=False):
    try:
        freeze_seeds(int(config.get('seed', 42)))
        process_photo = photo and not files_processed(output_file, config, "photo.vtp", save_landmarks=landmark)
        process_raw = (
            config["process_raw_photo"]
            and photoraw
            and not files_processed(
                os.path.join(output_file, "raw"),
                config,
                "photo-raw.vtp",
                save_landmarks=landmark and not photo,
            )
        )

        if landmark and (process_photo or (process_raw and not photo)):
            landmarks = ReadPolyData(os.path.join(input_file, 'landmarks.vtp'))

        if process_photo:
            photo_mesh = ReadPolyData(os.path.join(input_file, 'photo.vtp'))
            photo_mesh = preprocess_mesh(photo_mesh, config)
            
            if landmark:
                graph = convert_to_graph(photo_mesh, Path(input_file).parent.name, use_texture=config['use_texture'], landmarks=landmarks)
            else:
                graph = convert_to_graph(photo_mesh, Path(input_file).parent.name, use_texture=config['use_texture'])
                
            if config["save_graphs"]:
                torch.save(graph, os.path.join(output_file, 'graph.pt'))
            if config["save_features"]:
                # Save graph-related features to disk for later reuse.
                torch.save(graph.x, os.path.join(output_file, "normals.pt"))
                torch.save(graph.pos, os.path.join(output_file, "coordinates.pt"))
                torch.save(graph.edge_weight, os.path.join(output_file,"edge_weights.pt"))

                # Optional fields (only saved if they exist on the graph)
                if hasattr(graph, "texture") and config['use_texture']:
                    torch.save(graph.texture, os.path.join(output_file, "textures.pt"))
                if hasattr(graph, "landmarks") and landmark:
                    torch.save(graph.landmarks, os.path.join(output_file, "landmarks.pt"))

            
            # Always write outputs for the meshes we processed
            if config["save_vtp_files"]:
                WritePolyData(photo_mesh, os.path.join(output_file, 'photo.vtp'))

                # Copy landmarks if they were provided
                if landmark:
                    # Save as VTP so downstream steps can reload consistently
                    WritePolyData(landmarks, os.path.join(output_file, 'landmarks.vtp'))

            del graph
            del photo_mesh
            release_memory()
        else:
            if photo:
                print(f'The photo {Path(input_file).name} has been processed......')
                
                
        if process_raw:
            output_raw_file = os.path.join(output_file,'raw')
            os.makedirs(output_raw_file, exist_ok=True)
            photo_raw_mesh = ReadPolyData(os.path.join(input_file, 'photo-raw.vtp'))
            photo_raw_mesh = preprocess_mesh(photo_raw_mesh, config)
            if landmark and not photo:
                graph_raw = convert_to_graph(photo_raw_mesh, Path(input_file).parent.name, use_texture=config['use_texture'], landmarks=landmarks)
            else:
                graph_raw = convert_to_graph(photo_raw_mesh, Path(input_file).parent.name, use_texture=config['use_texture'])
            
            if config["save_graphs"]:
                torch.save(graph_raw, os.path.join(output_raw_file, 'graph.pt'))
            if config["save_features"]:
                # Save graph-related features to disk for later reuse.
                torch.save(graph_raw.x, os.path.join(output_raw_file, "normals.pt"))
                torch.save(graph_raw.pos, os.path.join(output_raw_file, "coordinates.pt"))
                torch.save(graph_raw.edge_weight, os.path.join(output_raw_file,"edge_weights.pt"))

                # Optional fields (only saved if they exist on the graph)
                if hasattr(graph_raw, "texture") and config['use_texture']:
                    torch.save(graph_raw.texture, os.path.join(output_raw_file, "textures.pt"))
                if hasattr(graph_raw, "landmarks") and (landmark and not photo):
                    torch.save(graph_raw.landmarks, os.path.join(output_raw_file, "landmarks.pt"))
            if config["save_vtp_files"]:
                WritePolyData(photo_raw_mesh, os.path.join(output_raw_file, 'photo-raw.vtp'))

            del graph_raw
            del photo_raw_mesh
            release_memory()
        else:
            if photoraw:
                print(f'The photo-raw {Path(input_file).name} has been processed......')
    finally:
        if "landmarks" in locals():
            del landmarks
        release_memory()
    # return graph
