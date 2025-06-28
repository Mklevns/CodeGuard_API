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
- **API Key Authentication implemented**: Bearer token authentication added to secure the /audit endpoint
- Authentication module created with secure token validation and timing attack protection
- New /auth/status endpoint for API key verification
- OpenAPI spec updated with security schemas for GPT Action compatibility
- **Legal compliance for OpenAI GPT Actions**: Privacy policy and terms of service documents created
- Privacy policy endpoint /privacy-policy serving comprehensive data protection information
- Terms of service endpoint /terms-of-service covering usage rights and responsibilities
- Both documents designed for OpenAI GPT Actions integration requirements
- **June 28, 2025**: Fixed Pydantic v2 compatibility warnings and authentication issues
- Updated Field definitions to use json_schema_extra instead of deprecated schema_extra
- Fixed authentication to allow requests without headers in development mode
- Authentication now works properly for automated tools and OpenAI GPT Actions
- **Cloud Run deployment fixes applied**: Fixed run command to use main.py explicitly instead of $file variable
- Enhanced root endpoint (/) to include "status": "healthy" for faster health check responses
- Updated application startup to properly handle PORT environment variable and environment detection
- Fixed uvicorn reload warning by using import string format in development mode
- Production entry point (run.py) confirmed working for Cloud Run deployment
- **OpenAPI deployment endpoint added**: Created `/openapi-deployment.yaml` endpoint to serve production OpenAPI spec
- Fixed 404 errors for users accessing OpenAPI specification from deployed application
- API key authentication fully functional and tested with production security enabled
- **Deployment crash fixes applied**: Created simplified `main_production.py` entry point for reliable Cloud Run deployment
- Fixed exit status 126 errors by ensuring proper file permissions and simplified startup process
- Updated startup script to use minimal configuration for better deployment reliability

## Deployment Status

**Current Status**: ✓ Live and operational
**Production HTTPS URL**: https://87ee31f3-2ea8-47fa-bfc6-dab95a535424-00-2j7aj3sdppcjx.riker.replit.dev
**Key Endpoints**:
- `/audit` - Main code analysis endpoint (POST) - Requires API key
- `/auth/status` - Authentication verification endpoint 
- `/docs` - Interactive API documentation 
- `/health` - Service health check
- `/.well-known/openapi.yaml` - OpenAPI spec for GPT Actions

## Changelog

- June 27, 2025: Initial setup and full implementation completed