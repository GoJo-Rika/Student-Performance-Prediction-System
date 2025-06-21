from pathlib import Path
from typing import List

from setuptools import find_packages, setup

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with Path(file_path).open() as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="student-performance-end-to-end-project",
    version="0.0.1",
    author="MitPatel",
    author_email="patel.m9521@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
