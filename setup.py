from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Reads a requirements file and returns a list of valid package requirements.
    Excludes lines like '-e .' which are not valid in `install_requires`.
    """
    requirements = []
    try:
        with open(file_path, 'r') as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('-e')]
    except FileNotFoundError:
        raise FileNotFoundError(f"Requirements file not found: {file_path}")
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Utkarsh',
    author_email='utkarshkarki97@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
