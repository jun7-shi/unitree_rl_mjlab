"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
]

EXTRAS_REQUIRE = {
    "webcam": [
        "numpy>=1.26,<2",
        "opencv-contrib-python>=4.8,<4.12",
        "mediapipe>=0.10.14,<0.11",
    ],
}

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
