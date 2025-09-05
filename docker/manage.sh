#!/bin/bash

# Docker Management Scripts for Cyber-Physical Systems
# ===================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================${NC}"
}

# Build Docker images
build() {
    print_header "Building Docker Images"
    
    print_status "Building cyber-physical systems image..."
    docker build -t cyber-physical-systems:latest .
    
    print_status "Build completed successfully!"
}

# Start services (development mode)
start() {
    print_header "Starting Development Environment"
    
    print_status "Starting services with docker-compose..."
    docker-compose up -d mlflow-server data-collector
    
    print_status "Waiting for MLflow server to be ready..."
    sleep 10
    
    print_status "Starting model training..."
    docker-compose up model-trainer
    
    print_status "Starting prediction API..."
    docker-compose up -d prediction-api
    
    print_status "Development environment started!"
    print_status "MLflow UI: http://localhost:5000"
    print_status "Prediction API: http://localhost:8080"
}

# Start production environment
start_prod() {
    print_header "Starting Production Environment"
    
    print_status "Starting production services with PostgreSQL..."
    docker-compose --profile production up -d
    
    print_status "Production environment started!"
    print_status "MLflow UI: http://localhost:5000"
    print_status "Prediction API: http://localhost:8080"
}

# Stop services
stop() {
    print_header "Stopping Services"
    
    print_status "Stopping all containers..."
    docker-compose down
    
    print_status "Services stopped!"
}

# Clean up everything
clean() {
    print_header "Cleaning Up Docker Resources"
    
    print_warning "This will remove all containers, images, and volumes!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Stopping containers..."
        docker-compose down -v
        
        print_status "Removing images..."
        docker rmi cyber-physical-systems:latest 2>/dev/null || true
        
        print_status "Removing unused Docker resources..."
        docker system prune -f
        
        print_status "Cleanup completed!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Show logs
logs() {
    service=${1:-""}
    
    if [ -z "$service" ]; then
        print_status "Showing logs for all services..."
        docker-compose logs -f
    else
        print_status "Showing logs for $service..."
        docker-compose logs -f "$service"
    fi
}

# Show status
status() {
    print_header "Container Status"
    docker-compose ps
    
    print_header "Service Health"
    
    # Check MLflow
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        print_status "MLflow Server: ✓ Running"
    else
        print_warning "MLflow Server: ✗ Not accessible"
    fi
    
    # Check Prediction API
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        print_status "Prediction API: ✓ Running"
    else
        print_warning "Prediction API: ✗ Not accessible"
    fi
}

# Run tests
test() {
    print_header "Running Tests"
    
    print_status "Building test image..."
    docker build -t cyber-physical-test:latest -f Dockerfile.test .
    
    print_status "Running tests..."
    docker run --rm cyber-physical-test:latest
    
    print_status "Tests completed!"
}

# Backup data
backup() {
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_dir="backups/backup_$timestamp"
    
    print_header "Creating Backup"
    
    print_status "Creating backup directory: $backup_dir"
    mkdir -p "$backup_dir"
    
    print_status "Backing up MLflow data..."
    cp -r mlruns "$backup_dir/"
    
    print_status "Backing up application data..."
    cp -r data "$backup_dir/"
    
    print_status "Creating archive..."
    tar -czf "$backup_dir.tar.gz" "$backup_dir"
    rm -rf "$backup_dir"
    
    print_status "Backup created: $backup_dir.tar.gz"
}

# Show help
help() {
    echo "Docker Management Script for Cyber-Physical Systems"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build      Build Docker images"
    echo "  start      Start development environment"
    echo "  start-prod Start production environment"
    echo "  stop       Stop all services"
    echo "  clean      Clean up Docker resources"
    echo "  logs       Show logs [service_name]"
    echo "  status     Show container status and health"
    echo "  test       Run tests"
    echo "  backup     Create backup of data"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 start"
    echo "  $0 logs mlflow-server"
    echo "  $0 status"
}

# Main script logic
case "${1:-help}" in
    build)
        build
        ;;
    start)
        start
        ;;
    start-prod)
        start_prod
        ;;
    stop)
        stop
        ;;
    clean)
        clean
        ;;
    logs)
        logs "$2"
        ;;
    status)
        status
        ;;
    test)
        test
        ;;
    backup)
        backup
        ;;
    help|*)
        help
        ;;
esac
