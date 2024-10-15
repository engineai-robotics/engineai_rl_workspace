import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import argparse
from engineai_rl_workspace.utils import get_args
from engineai_rl_workspace import ENGINEAI_WORKSPACE_ROOT_DIR


def run_command(
    run_name,
    play_script_path,
    play_command_template,
    export_policy_script_path,
    export_policy_command_template,
):
    # Construct the command with the current folder name
    play_command = play_command_template.format(
        script=play_script_path, load_run=run_name
    )
    export_policy_command = export_policy_command_template.format(
        script=export_policy_script_path, load_run=run_name
    )

    # Print the command to be executed (for debugging purposes)
    print(f"Running command: {play_command}\n")

    # Execute the play_command
    subprocess.run(play_command, shell=True)

    # Execute the export_policy_command
    subprocess.run(export_policy_command, shell=True)


def run_experiment(
    parent_dir,
    play_script_name,
    export_policy_script_name,
    exp_name,
    sub_exp_name,
    skip_exist,
    record_length,
    headless,
    least_iteration,
):
    # Get the full path to the script
    script_path = os.path.join(os.path.dirname(__file__), play_script_name)
    export_policy_script_path = os.path.join(
        os.path.dirname(__file__), export_policy_script_name
    )

    # List all subdirectories in the parent directory
    runs = [
        f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))
    ]

    headless_option = "--headless" if headless else ""
    # Define the command template
    play_command_template = f"python {{script}} --exp_name={exp_name} --sub_exp_name={sub_exp_name} --load_run={{load_run}} --video --record_length={record_length} {headless_option}"
    export_policy_command_template = f"python {{script}} --exp_name={exp_name} --sub_exp_name={sub_exp_name} --load_run={{load_run}}"
    # Use ThreadPoolExecutor to manage concurrent execution
    with ThreadPoolExecutor() as executor:
        filtered_runs = []
        for run in runs:
            run_dir = os.path.join(parent_dir, run)
            models = [
                file
                for file in os.listdir(os.path.join(run_dir, "checkpoints"))
                if "model" in file
            ]
            models.sort(key=lambda m: f"{m:0>15}")
            lastest_model = int(models[-1][6:-3])
            if lastest_model >= least_iteration:
                if skip_exist:
                    if (
                        "metrics" not in os.listdir(run_dir)
                        or "play_videos" not in os.listdir(run_dir)
                        or "policies" not in os.listdir(run_dir)
                    ):
                        filtered_runs.append(run)
                else:
                    filtered_runs.append(run)
        # Submit tasks to the executor
        futures = [
            executor.submit(
                run_command,
                run,
                script_path,
                play_command_template,
                export_policy_script_path,
                export_policy_command_template,
            )
            for run in filtered_runs
        ]

        # Wait for all futures to complete
        for future in futures:
            # Optionally handle exceptions here
            future.result()


if __name__ == "__main__":
    args = get_args()
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run plays and export policies.")
    parser.add_argument(
        "--skip_exist", action="store_true", help="Skip runs already run before."
    )
    parser.add_argument(
        "--least_iteration",
        type=int,
        default=10000,
        help="Filter out runs with less than the least iteration",
    )

    # Parse the arguments
    extra_args, unknown = parser.parse_known_args()

    root_dir = os.path.join(
        ENGINEAI_WORKSPACE_ROOT_DIR, "logs", args.exp_name, args.sub_exp_name
    )
    # Run the experiment
    run_experiment(
        root_dir,
        "play.py",
        "export_policy.py",
        args.exp_name,
        args.sub_exp_name,
        extra_args.skip_exist,
        args.record_length,
        args.headless,
        extra_args.least_iteration,
    )
