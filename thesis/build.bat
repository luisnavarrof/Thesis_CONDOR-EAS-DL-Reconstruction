@echo off
rem Build main.pdf: pdflatex -> bibtex -> pdflatex x2.
rem Usage: build.bat          compile
rem        build.bat clean    remove auxiliary files
setlocal
cd /d "%~dp0"

if /i "%1"=="clean" goto clean

pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul || goto fail
bibtex main >nul || goto fail
pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul || goto fail
pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul || goto fail
echo main.pdf built.
goto end

:fail
echo Build failed; see main.log / main.blg.
exit /b 1

:clean
del /q main.aux main.bbl main.blg main.lof main.lot main.out main.toc main.log 2>nul
echo Cleaned.

:end
endlocal
