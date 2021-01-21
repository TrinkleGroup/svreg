import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="svreg", # Replace with your own username
    version="0.0.1",
    author="Josh Vita",
    author_email="vita.joshua@gmail.com",
    description='A package for constructing spline-based interatomic potentials using symbolic regression',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TrinkleGroup/svreg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)