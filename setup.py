import os
from setuptools import find_packages, setup

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE, "r") as f:
    requires = f.readlines()
    install_requires = requires[:requires.index("EXTRAS_REQUIRE\n")]
    extras_require = {}
    extra = ""
    for extra_require in requires[requires.index("EXTRAS_REQUIRE\n") + 1:]:
        if extra_require.startswith("-"):
            extra = extra_require.strip().lstrip("-")
            continue
        if extra in extras_require:
            extras_require[extra].append(extra_require)
        else:
            extras_require[extra] = [extra_require]

    
setup(
    name="src",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require, # Vacío porque seguí un template para crear esto
    version="0.1.0",
)