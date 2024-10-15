import pandas as pd
import json
from pathlib import Path


def csv_to_motion_json(csv_filepath):
    """Convert a single CSV file to motion JSON format"""
    try:
        # Read the first line separately to get frame_duration
        with open(csv_filepath) as f:
            first_line = f.readline()
            frame_duration = float(first_line.split(",")[1])

        # Read the rest of the CSV
        df = pd.read_csv(csv_filepath, skiprows=1)

        # Create the base JSON structure
        motion_data = {
            "LoopMode": "Wrap",
            "FrameDuration": frame_duration,
            "EnableCycleOffsetPosition": True,
            "EnableCycleOffsetRotation": True,
            "MotionWeight": 1,
            "Frames": [],
        }

        # Convert each row to a frame array
        for _, row in df.iterrows():
            # Convert row to list and remove NaN values
            frame = [float(x) for x in row if pd.notna(x)]
            motion_data["Frames"].append(frame)

        # Write to JSON file
        output_path = str(csv_filepath).rsplit(".", 1)[0] + ".json"
        with open(output_path, "w") as f:
            json.dump(motion_data, f, indent=2)

        return (True, f"Successfully converted {csv_filepath.name}")
    except Exception as e:
        return (False, f"Error converting {csv_filepath.name}: {str(e)}")


def csv_to_motion_json_for_folder(folder_path):
    """Convert all CSV files in a folder to JSON format"""
    # Convert string path to Path object
    folder_path = Path(folder_path)

    # Ensure the folder exists
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return

    # Get all CSV files in the folder
    csv_files = list(folder_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return

    print(f"Found {len(csv_files)} CSV files to convert")

    # Process each file
    successful = 0
    failed = 0

    for csv_file in csv_files:
        try:
            success, message = csv_to_motion_json(csv_file)
            print(message)
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
            failed += 1

    # Print summary
    print("\nConversion Summary:")
    print(f"Total files processed: {len(csv_files)}")
    print(f"Successfully converted: {successful}")
    print(f"Failed conversions: {failed}")
