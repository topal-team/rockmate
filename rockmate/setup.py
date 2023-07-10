from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            "rockmate.csolver",
            ["src/solver.c"],
            extra_compile_args=["-O3", "--std=c99"],
        ),
    ],
)

