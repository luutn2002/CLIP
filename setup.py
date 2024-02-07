import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="mapleiqa",
    py_modules=["mapleiqa"],
    version="0.0.1",
    description="MaPLe models for blindd image quality assessment (BIQA).",
    author="LUU Trong Nhan",
    author_email = "ltnhan0902@gmail.com",
    packages=find_packages(exclude=["examples*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True
)