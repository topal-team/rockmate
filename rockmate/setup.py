from distutils.core import setup, Extension
setup(
    ext_modules=[
        Extension(
            "rockmate.solvers.csolver",
            ["rockmate/solvers/solver.c"],
            extra_compile_args=["-O3", "--std=c99"],
        ),
    ],
)

