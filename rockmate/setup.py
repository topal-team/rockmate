from distutils.core import setup, Extension
setup(
    ext_modules=[
        Extension(
            "rockmate.solvers.rk_rotor.csolver",
            ["src/rockmate/solvers/rk_rotor/solver.c"],
            extra_compile_args=["-O3", "--std=c99"],
        ),
    ],
)

