@echo off
REM Windows Docker Management Script for Cyber-Physical Systems
REM ============================================================

setlocal enabledelayedexpansion

REM Function to print colored output (Windows)
:print_status
echo [INFO] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

:print_header
echo ======================================
echo %~1
echo ======================================
goto :eof

REM Build Docker images
:build
call :print_header "Building Docker Images"
call :print_status "Building cyber-physical systems image..."
docker build -t cyber-physical-systems:latest .
if %ERRORLEVEL% NEQ 0 (
    call :print_error "Build failed!"
    exit /b 1
)
call :print_status "Build completed successfully!"
goto :eof

REM Start services (development mode)
:start
call :print_header "Starting Development Environment"
call :print_status "Starting services with docker-compose..."
docker-compose up -d mlflow-server data-collector
if %ERRORLEVEL% NEQ 0 (
    call :print_error "Failed to start services!"
    exit /b 1
)

call :print_status "Waiting for MLflow server to be ready..."
timeout /t 10 /nobreak >nul

call :print_status "Starting model training..."
docker-compose up model-trainer

call :print_status "Starting prediction API..."
docker-compose up -d prediction-api

call :print_status "Development environment started!"
call :print_status "MLflow UI: http://localhost:5000"
call :print_status "Prediction API: http://localhost:8080"
goto :eof

REM Start production environment
:start_prod
call :print_header "Starting Production Environment"
call :print_status "Starting production services with PostgreSQL..."
docker-compose --profile production up -d
call :print_status "Production environment started!"
call :print_status "MLflow UI: http://localhost:5000"
call :print_status "Prediction API: http://localhost:8080"
goto :eof

REM Stop services
:stop
call :print_header "Stopping Services"
call :print_status "Stopping all containers..."
docker-compose down
call :print_status "Services stopped!"
goto :eof

REM Clean up everything
:clean
call :print_header "Cleaning Up Docker Resources"
call :print_warning "This will remove all containers, images, and volumes!"
set /p choice="Are you sure? (y/N): "
if /i "!choice!"=="y" (
    call :print_status "Stopping containers..."
    docker-compose down -v
    
    call :print_status "Removing images..."
    docker rmi cyber-physical-systems:latest 2>nul
    
    call :print_status "Removing unused Docker resources..."
    docker system prune -f
    
    call :print_status "Cleanup completed!"
) else (
    call :print_status "Cleanup cancelled."
)
goto :eof

REM Show logs
:logs
if "%~2"=="" (
    call :print_status "Showing logs for all services..."
    docker-compose logs -f
) else (
    call :print_status "Showing logs for %~2..."
    docker-compose logs -f %~2
)
goto :eof

REM Show status
:status
call :print_header "Container Status"
docker-compose ps

call :print_header "Service Health"
REM Check MLflow
curl -s http://localhost:5000 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    call :print_status "MLflow Server: ✓ Running"
) else (
    call :print_warning "MLflow Server: ✗ Not accessible"
)

REM Check Prediction API
curl -s http://localhost:8080/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    call :print_status "Prediction API: ✓ Running"
) else (
    call :print_warning "Prediction API: ✗ Not accessible"
)
goto :eof

REM Backup data
:backup
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,8%_%dt:~8,6%"
set "backup_dir=backups\backup_%timestamp%"

call :print_header "Creating Backup"
call :print_status "Creating backup directory: %backup_dir%"
mkdir "%backup_dir%" 2>nul

call :print_status "Backing up MLflow data..."
xcopy /E /I mlruns "%backup_dir%\mlruns\" >nul

call :print_status "Backing up application data..."
xcopy /E /I data "%backup_dir%\data\" >nul

call :print_status "Backup created: %backup_dir%"
goto :eof

REM Show help
:help
echo Docker Management Script for Cyber-Physical Systems
echo.
echo Usage: %~nx0 ^<command^> [options]
echo.
echo Commands:
echo   build      Build Docker images
echo   start      Start development environment
echo   start-prod Start production environment
echo   stop       Stop all services
echo   clean      Clean up Docker resources
echo   logs       Show logs [service_name]
echo   status     Show container status and health
echo   backup     Create backup of data
echo   help       Show this help message
echo.
echo Examples:
echo   %~nx0 build
echo   %~nx0 start
echo   %~nx0 logs mlflow-server
echo   %~nx0 status
goto :eof

REM Main script logic
if "%~1"=="" goto help
if "%~1"=="build" goto build
if "%~1"=="start" goto start
if "%~1"=="start-prod" goto start_prod
if "%~1"=="stop" goto stop
if "%~1"=="clean" goto clean
if "%~1"=="logs" goto logs
if "%~1"=="status" goto status
if "%~1"=="backup" goto backup
if "%~1"=="help" goto help
goto help
