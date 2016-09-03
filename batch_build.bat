set proj_path="%~dp0\modelBER_CUDA"
set msbuild="C:\Program Files (x86)\MSBuild\12.0\Bin\MSBuild.exe"

cd /d %1
mkdir src
copy %proj_path% src

for %%i in (*.cuh) do call :build "%%i"
pause

goto :eof

:build

echo #include "%~f1" > src\modelBER_params_link.cuh
%msbuild% src\modelBER_CUDA.vcxproj /t:Build /p:Configuration=Release /p:OutDir="%cd%" /p:TargetName=%~n1

GOTO :EOF