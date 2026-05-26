import argparse
import os
import yaml
import sys
from pathlib import Path

# Ensure `src` is importable when running this script directly.
# This works even when current working directory is the repo root.
sys.path.append(str(Path(__file__).resolve().parents[1]))


from src.pipeline import preprocess_pipeline
from src.io import FindFilesToProcess



def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_args_config(args, config):
    """
    Permite sobreescribir el YAML desde CLI si se desea
    """
    config = config.copy()

    if args.input:
        config["input_file"] = args.input

    if args.output:
        config["output_file"] = args.output

    return config


def validate_config(config):
    required = ["input_dir", "output_dir"]

    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    if not os.path.exists(config["input_dir"]):
        raise ValueError(f"Input file not found: {config['input_dir']}")


def run_test(config, result):
    """
    Test to verify the outputs
    """
    print("\nRunning basic test...")

    # test 1: archivo generado
    if os.path.exists(config["output_file"]):
        print("✅ Output file created")
    else:
        print("❌ Output file NOT created")

    # test 2: graph generado
    if result is not None:
        if "pos" in result and "x" in result:
            print("✅ Graph structure OK")
        else:
            print("⚠️ Graph missing fields")
    else:
        print("❌ No graph returned")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="configs/test.yaml")
    parser.add_argument("--input", required=False)
    parser.add_argument("--output", required=False)
    parser.add_argument('--save_df', default=False)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    # Load YAML config
    config = load_config(args.config)

    # Merge CLI args
    config = merge_args_config(args, config)
    
    df = FindFilesToProcess(config["input_dir"], verbose=True)

    # # Validate
    validate_config(config)

    print("Running with configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
        
    df['status'] = ''
    df['error'] = ''
    if not os.path.exists(config['output_dir']):
        os.mkdir(config['output_dir'])
    # # Run processing
    for idx, row in df.iterrows():
        folder_path = row["folder_path"]
        folder_name = row["folder_names"]
        parent_folder = row["parent_names"]
        photo = row["has_photo"]
        landmarks= row["has_landmarks"]
        photoraw = row["has_photo_raw"]
        output_parent_folder = Path(config["output_dir"]) / parent_folder
        if not os.path.exists(output_parent_folder):
            os.mkdir(output_parent_folder)
            
        
        print(f"\n--- Processing: {folder_name} ---")
        try:
            # Definir output por sujeto
            output_file = output_parent_folder / folder_name
            if not os.path.exists(output_file):
                os.mkdir(output_file)

            # Ejecutar pipeline
            preprocess_pipeline(folder_path, output_file, config, photo, landmarks, photoraw)
            
            mask = (df["folder_names"] == folder_name) & (df["parent_names"] == row["parent_names"])
            df.loc[mask, "status"] = "success"

        except Exception as e:
            print(f"❌ Error processing {folder_name}: {e}")
            mask = (df["folder_names"] == folder_name) & (df["parent_names"] == row["parent_names"])
            df.loc[mask, "status"] = "failed"
            df.loc[mask, "error"] = str(e)
            
    if config["save_df"]:
        df.to_excel(os.path.join(config["output_dir"], "Data_preprocessed.xlsx"), index=False)


if __name__ == "__main__":
    main()
