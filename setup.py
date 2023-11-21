import codecs
import os

from setuptools import find_packages, setup

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")

def get_requirements():
    with codecs.open(REQUIREMENTS_FILE) as buff:
        return buff.read().splitlines()
    
setup(
    name="src",
    packages=find_packages(),
    install_requires=get_requirements(), # Vacío porque seguí un template para crear esto
    version="0.1.0",
)