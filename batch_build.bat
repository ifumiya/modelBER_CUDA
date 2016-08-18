set sol_path="C:\Users\inuka\Documents\Inukai\code\modelBER_CUDA"
set proj_path=%sol_path%\modelBER_CUDA
set msbuild="C:\Program Files (x86)\MSBuild\12.0\Bin\MSBuild.exe"

cd /d %1

for %%i in (*.cuh) do call :build "%%i"


goto :eof

:build

ren %proj_path%\modelBER_params_link.cuh modelBER_params_link.cuh.bak
echo #include "%~f1" > %proj_path%\modelBER_params_link.cuh


:del %sol_path%\Release\modelBER_CUDA.exe
%msbuild% %proj_path%\modelBER_CUDA.vcxproj /t:Build;Run /p:Configuration=Release /p:OutDir="%cd%" /p:TargetName=%~n1
rem Build Clean Run

del %proj_path%\modelBER_params_link.cuh
ren %proj_path%\modelBER_params_link.cuh.bak modelBER_params_link.cuh

GOTO :EOF