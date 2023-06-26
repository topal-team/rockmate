from distutils.core import setup, Extension
setup(
    ext_modules=[
        Extension(
            "rockmate.csolver",
            ["rockmate/solver.c"],
            extra_compile_args=["-O3", "--std=c99"],
        ),
    ],
)

