from setuptools import setup, Extension, find_packages

packages = find_packages()


def run_setup():
    extensions = [Extension('dynamic_programs',
                            sources = ['rotor/algorithms/dynamic_programs.c'])]
    

    setup (name = 'rotor',
           version = '0.1',
           description = 'Rematerialize Optimally with pyTORch',
           install_requires = ['psutil'],
           packages = packages,
           ext_modules = extensions)


run_setup()
