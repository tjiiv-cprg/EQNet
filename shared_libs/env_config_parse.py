import yaml


def extend_extra_sys_path(env_config_yaml_file_path):
    with open(env_config_yaml_file_path, 'r') as f:
        env_config = yaml.safe_load(f)
    extra_sys_paths = env_config['extra_sys_path']
    import sys
    sys.path.extend(extra_sys_paths)


from git import Repo, GitCommandError

def check_git_status_and_get_branch():
    """
    Check if the current working directory is a Git repository and if it is clean.

    Returns the name of the current branch if the repository is clean, otherwise returns None.

    Returns:
        str or None: The name of the current branch if the repository is clean, otherwise None.
    """
    try:
        # Initialize a Repo object for the current directory
        repo = Repo('.')
        if not repo.bare:
            # Check if the repository is clean
            if repo.is_dirty(untracked_files=True):
                print("The current repository has uncommitted changes.")
                return None
            else:
                # Get the current branch name
                current_branch = repo.active_branch.name
                # print(f"The current repository is clean. Current branch: {current_branch}")
                return current_branch
        else:
            print("This is a bare repository.")
            return None
    except GitCommandError as e:
        print(f"Git command error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage:
# branch_name = check_git_status_and_get_branch()
# if branch_name:
#     print(f"Current clean branch: {branch_name}")
# else:
#     print("The working directory is not clean or is not a Git repository.")