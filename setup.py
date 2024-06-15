from setuptools import find_namespace_packages, setup, find_packages


with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [req.rstrip("\n") for req in fh]

setup(
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    # packages=["backend", "pipelines", "model"],
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    package_data= {
        # all .dat files at any package depth
        "model.EfLipReading.configs": ['**/*.ini', '**/*.yml', ],
        "model.EfLipReading.model.labels": ['**/*.ini', '**/*.yml', ],
        "model.EfLipReading.data.detectors.mediapipe": ['**/*.ini', '**/*.yml', '**/*.npy'],
        "pipelines.detectors.mediapipe": ['**/*.ini', '**/*.yml', '**/*.npy'],
        # into the data folder (being into a module) but w/o the init file
    },
    python_requires=">=3.8",
    install_requires=requirements,
)