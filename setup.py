from setuptools import find_namespace_packages, setup, find_packages


with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [req.rstrip("\n") for req in fh]

setup(
    name="backending",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    # packages=["backend", "pipelines", "model"],
    packages=find_packages(),
    package_dir={"pipelines": "./pipelines", 'model_zoo': './model_zoo'},
    include_package_data=True,
    package_data= {
        # all .dat files at any package depth
        "pipelines.detectors.mediapipe": ['**/*.yaml', '**/*.npy', '*.npy'],
        "model_zoo": ['**/*.ckpt', '**/**/*.ckpt', '*.ckpt'],
        # into the data folder (being into a module) but w/o the init file
    },
    python_requires=">=3.8",
    install_requires=requirements,
)