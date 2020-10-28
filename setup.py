from setuptools import setup, find_packages

setup(
    name="simplectf",
    version="0.0.1",
    description="Setting up a python package",
    author="Albert Xu",
    author_email="albert.t.xu@gmail.com",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=["mrcfile", "numpy", "scipy"],
    entry_points={"console_scripts": ["simplectf=simplectf.__main__:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
