from distutils.core import setup, Extension

dynamic_programs = Extension('dynamic_programs',
                     sources = ['dynamic_programs.c'])

setup (name = 'dynamic_programs',
       version = '1.0',
       description = 'C implementation of Dynamic Programs',
       ext_modules = [dynamic_programs])
