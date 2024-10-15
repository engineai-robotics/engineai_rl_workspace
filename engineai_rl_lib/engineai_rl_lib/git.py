import os
import git
import pathlib


def store_code_state(logdir, repository_file_path) -> list:
    try:
        repo = git.Repo(repository_file_path, search_parent_directories=True)
    except Exception:
        print(f"Could not find git repository in {repository_file_path}. Skipping.")
        # skip if not a git repository
        return
    # get the name of the repository
    repo_name = pathlib.Path(repo.working_dir).name
    git_info_file = os.path.join(logdir, "git_info.txt")
    # check if the diff file already exists
    if os.path.isfile(git_info_file):
        return
    # write the diff file
    with open(git_info_file, "x", encoding="utf-8") as f:
        content = ["--- Git Repository Information ---", f"Repository: {repo_name}"]
        current_commit = repo.head.commit
        content.extend(
            [
                f"\n--- Current Commit ---",
                f"Hash: {current_commit.hexsha}",
                f"Short Hash: {current_commit.hexsha[:8]}",
                f"Author: {current_commit.author}",
                f"Date: {current_commit.authored_datetime}",
                f"Message: {current_commit.message.strip()}",
            ]
        )
        f.write("\n".join(content))
