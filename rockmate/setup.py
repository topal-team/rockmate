from distutils.core import setup, Extension
setup(
    ext_modules=[
        Extension(
            "rockmate.solvers.csolver",
            ["src/solvers/solver.c"],
            extra_compile_args=["-O3", "--std=c99"],
        ),
    ],
)

