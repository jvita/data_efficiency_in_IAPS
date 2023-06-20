from setuptools import setup, find_packages

setup(
    name = "ip_explorer",
    version = "0.0.1",
    author = "Josh Vita",
    description = ("A package for facilitating model exploration"),
    license = "BSD",
    keywords = "interatomic potentials potential energy surfaces loss landscapes",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires = [
    ],
)
