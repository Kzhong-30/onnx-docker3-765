#!/bin/bash

set -e

echo "=========================================="
echo "Image Classification API - Run Tests"
echo "=========================================="
echo ""

if ! command -v docker &> /dev/null; then
    echo "Docker is not installed or not in PATH"
    echo "Please install Docker Desktop and try again"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Docker Compose is not installed or not in PATH"
    echo "Please install Docker Compose and try again"
    exit 1
fi

echo "Building and starting test container..."
echo ""

COMPOSE_CMD="docker-compose"
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
fi

$COMPOSE_CMD up test --build

echo ""
echo "=========================================="
echo "Tests completed!"
echo "HTML Report: test_reports/test_report.html"
echo "=========================================="
