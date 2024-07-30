import yaml
import re

def parse_markdown_yaml_metadata(file_path: str) -> dict:
    """
    Parses the YAML metadata from a Markdown file.

    This function reads a Markdown file and extracts the YAML metadata block
    at the beginning of the file. The YAML block is expected to be enclosed
    within two sets of triple hyphens ('---').

    Args:
        file_path (str): The path to the Markdown file to parse.

    Returns:
        dict: A dictionary containing the parsed YAML metadata.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If the YAML content is not valid.

    Example:
        >>> metadata = parse_markdown_yaml_metadata('example.md')
        >>> print(metadata)
        {'title': 'Example', 'author': 'Author Name', 'date': 'YYYY-MM-DD'}
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Regex to match the YAML block at the start of the file
            yaml_block_match = re.search(r'^---\s*\n(.+?)\s*\n---', content, re.DOTALL)
            if yaml_block_match:
                yaml_block = yaml_block_match.group(1)
                # Parse the YAML block into a dictionary
                metadata = yaml.safe_load(yaml_block)
                return metadata
            else:
                # Return an empty dictionary if no YAML block is found
                return {}
    except FileNotFoundError as e:
        # Raise the FileNotFoundError with a custom message
        raise FileNotFoundError(f"The file {file_path} does not exist.") from e
    except yaml.YAMLError as e:
        # Raise the YAMLError with a custom message
        raise yaml.YAMLError(f"Error parsing YAML in file {file_path}: {e}") from e

# Example usage:
# try:
#     metadata = parse_markdown_yaml_metadata('path_to_your_markdown_file.md')
#     print(metadata)
# except Exception as e:
#     print(f"An error occurred: {e}")


def extend_extra_sys_path(markdown_eec_file_path):
    extra_sys_paths = parse_markdown_yaml_metadata(markdown_eec_file_path)['extra_sys_path']
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