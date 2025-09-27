#!/usr/bin/env python3
"""
Simple Local Deployment Checker
Checks basic prerequisites for deployment without external dependencies
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and return success status and output"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def check_docker():
    """Check if Docker is available and running"""
    print("Checking Docker...")
    
    success, stdout, stderr = run_command("docker --version")
    if success:
        print(f"  Docker found: {stdout}")
    else:
        print(f"  ERROR: Docker not available")
        return False
    
    success, stdout, stderr = run_command("docker ps")
    if success:
        print("  Docker daemon is running")
        return True
    else:
        print("  ERROR: Docker daemon not running - Please start Docker Desktop")
        return False

def check_docker_compose():
    """Check if Docker Compose is available"""
    print("Checking Docker Compose...")
    
    success, stdout, stderr = run_command("docker-compose --version")
    if success:
        print(f"  Docker Compose found: {stdout}")
        return True
    else:
        print("  ERROR: Docker Compose not available")
        return False

def check_files():
    """Check if required files exist"""
    print("Checking required files...")
    
    required_files = [
        'docker-compose.yml',
        'Dockerfile', 
        'requirements.txt',
        'docker/prediction_api.py',
        'ml-models/basic_forecaster.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  Found: {file_path}")
        else:
            print(f"  ERROR: Missing {file_path}")
            all_exist = False
    
    return all_exist

def show_deployment_commands():
    """Show the commands needed for deployment"""
    print("\nDeployment Commands:")
    print("=" * 30)
    print("1. Build containers:")
    print("   docker-compose build")
    print("\n2. Start services:")
    print("   docker-compose up -d")
    print("\n3. Check status:")
    print("   docker-compose ps")
    print("\n4. View logs:")
    print("   docker-compose logs")
    print("\n5. Access interfaces:")
    print("   MLflow: http://localhost:5000")
    print("   API: http://localhost:8080")
    print("   Health: http://localhost:8080/health")
    print("\n6. Stop services:")
    print("   docker-compose down")

def main():
    print("Local Deployment Prerequisites Check")
    print("=" * 40)
    
    # Check all prerequisites
    docker_ok = check_docker()
    compose_ok = check_docker_compose()
    files_ok = check_files()
    
    print("\nSummary:")
    print("=" * 20)
    if docker_ok and compose_ok and files_ok:
        print("All prerequisites satisfied - Ready for deployment!")
        show_deployment_commands()
        return True
    else:
        print("Some prerequisites are missing:")
        if not docker_ok:
            print("- Docker is not running")
        if not compose_ok:
            print("- Docker Compose not available")
        if not files_ok:
            print("- Required files missing")
        print("\nPlease fix these issues before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)