@ECHO OFF
SET ENV_FILE=environment.yml
SET ENV_NAME=tesina

:: Si existe, limpia el environment
ECHO Si es que ya existe, eliminando ambiente de conda...
CALL conda env remove --name %ENV_NAME% --yes

:: Crea el environment
ECHO Creando ambiente de conda...
CALL conda env create -f %ENV_FILE% || goto :error
CALL conda activate %ENV_NAME% || goto :error

:error
EXIT /b %errorlevel%
