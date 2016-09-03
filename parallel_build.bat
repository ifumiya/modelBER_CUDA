cd /d %~dp0

:check
if "%1"=="" goto :eof
start cmd /c batch_build "%1"

shift
goto check
