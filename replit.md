# CodeGuard API - Replit Development Guide

## Overview

CodeGuard is a FastAPI-based backend service that provides static code analysis for machine learning (ML) and reinforcement learning (RL) Python code. The service analyzes code files for syntax errors, style issues, and production concerns, returning structured audit results with fix suggestions. This service is designed to be consumed by OpenAI Actions via an OpenAPI specification.

## System Architecture

### Core Architecture
- **Framework**: FastAPI for modern, async API development with built-in OpenAPI support
- **Runtime**: Uvicorn ASGI server for production deployment
- **Language**: Python 3.10+ for compatibility with modern static analysis tools
- **Deployment**: Replit-hosted server exposing public HTTPS endpoints

### Analysis Engine
- **Primary Analyzer**: flake8 for syntax, style, and linting analysis
- **File Processing**: Temporary file system for secure code analysis
- **Security**: Input sanitization to prevent path traversal attacks

## Key Components

### 1. API Layer (`main.py`)
- **FastAPI Application**: Main application with CORS middleware for cross-origin requests
- **Endpoints**:
  - `GET /` - Service information and available endpoints
  - `POST /audit` - Main code analysis endpoint
- **OpenAPI Integration**: Automatic schema generation at `/openapi.json`

### 2. Data Models (`models.py`)
- **Request Models**:
  - `CodeFile`: Represents individual files with filename and content
  - `AuditOptions`: Optional configuration for analysis level, framework, and target platform
  - `AuditRequest`: Complete audit request with files and options
- **Response Models**:
  - `Issue`: Represents code issues with file, line, type, and description
  - `Fix`: Represents suggested fixes for issues
  - `AuditResponse`: Complete audit response with summary and results

### 3. Analysis Engine (`audit.py`)
- **Core Function**: `analyze_code()` processes audit requests
- **File Handling**: Secure temporary directory creation for analysis
- **Analysis Pipeline**: Integration with flake8 for static code analysis
- **Error Handling**: Graceful handling of file processing errors

## Data Flow

1. **Request Reception**: Client sends POST request to `/audit` with code files
2. **Input Validation**: Pydantic models validate request structure and content
3. **File Processing**: Files are written to temporary directory with sanitized names
4. **Static Analysis**: flake8 analyzes each file for issues
5. **Result Processing**: Issues are structured into response format
6. **Response Delivery**: JSON response with audit results and fix suggestions

## External Dependencies

### Python Packages
- **FastAPI**: Web framework for API development
- **Uvicorn**: ASGI server for production deployment
- **Pydantic**: Data validation and serialization
- **flake8**: Static code analysis tool

### Analysis Tools
- **flake8**: Primary static analysis engine for Python code
- **Future Extensions**: Designed to accommodate mypy, pylint, and libcst integration

## Deployment Strategy

### Replit Hosting
- **Platform**: Replit-hosted server for easy deployment and management
- **Endpoint**: Public HTTPS endpoint for external consumption
- **CORS**: Configured to allow cross-origin requests from any domain

### Production Considerations
- **Security**: Input sanitization and temporary file handling
- **Performance**: Temporary directory cleanup and resource management
- **Scalability**: Stateless design for horizontal scaling

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

- June 27, 2025: Initial CodeGuard API implementation completed and deployed
- API server running on port 5000 with public access at http://34.55.167.13:5000
- All core endpoints functional: /audit, /health, /.well-known/openapi.yaml, /docs
- Flake8 integration working correctly with comprehensive error detection
- OpenAPI specification updated with correct server URL for GPT Action integration

## Deployment Status

**Current Status**: ✓ Live and operational
**Public URL**: http://34.55.167.13:5000
**Key Endpoints**:
- `/audit` - Main code analysis endpoint (POST)
- `/docs` - Interactive API documentation 
- `/health` - Service health check
- `/.well-known/openapi.yaml` - OpenAPI spec for GPT Actions

## Changelog

- June 27, 2025: Initial setup and full implementation completed