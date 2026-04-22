@echo off
setlocal

set REPO_URL=https://github.com/bibun-kanou/anisoEngine2.git
set BRANCH=master

echo === RadianceLab Push Script ===
echo Repo: %REPO_URL%
echo.

:: Check if git is available
where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: git is not installed or not in PATH.
    pause
    exit /b 1
)

:: Initialize repo if needed
if not exist ".git" (
    echo Initializing git repository...
    git init
    git remote add origin %REPO_URL%
    echo Git repo initialized with remote: %REPO_URL%
) else (
    echo Git repo already initialized.
    :: Ensure remote is set correctly
    git remote set-url origin %REPO_URL% 2>nul || git remote add origin %REPO_URL%
)

:: Ensure we're on the right branch
git branch -M %BRANCH%

:: Stage all files (respects .gitignore)
echo.
echo Staging files...
git add -A

:: Show what will be committed
echo.
echo === Files staged ===
git status --short
echo.

:: Prompt for commit message
set /p COMMIT_MSG="Commit message (or press Enter for default): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Update: BDPT MIS, interior connections, KD-tree viz, PM fixes

:: Commit
echo.
echo Committing: %COMMIT_MSG%
git commit -m "%COMMIT_MSG%"

:: Push
echo.
echo Pushing to %REPO_URL% (%BRANCH%)...
git push -u origin %BRANCH%

if errorlevel 1 (
    echo.
    echo Push failed. If this is a new repo, try:
    echo   git push -u origin %BRANCH% --force
    echo.
    set /p FORCE="Force push? (y/N): "
    if /i "%FORCE%"=="y" (
        git push -u origin %BRANCH% --force
    )
)

echo.
echo === Done ===
