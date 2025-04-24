from setuptools import setup, find_packages

setup(
    name="tetris",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pygame>=2.5.2",
        "numpy>=1.24.0",
        "gymnasium>=0.29.1",
    ],
) 