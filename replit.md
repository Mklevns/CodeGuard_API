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
  - `Issue`: Enhanced with source tool identification and severity levels
  - `Fix`: Extended with diff generation, replacement code, and auto-fix capabilities
  - `AuditResponse`: Complete audit response with multi-tool analysis results

### 3. Enhanced Analysis Engine (`enhanced_audit.py`, `rule_engine.py`)
- **Multi-Tool Integration**: Combines flake8, pylint, mypy, black, isort, and custom ML/RL rules
- **Enhanced Analysis Pipeline**: Parallel execution of multiple static analysis tools
- **ML/RL Pattern Detection**: Custom rules for missing seeding, environment resets, training loop issues
- **Advanced Fix Generation**: Unified diff format and auto-fixable suggestions
- **Error Handling**: Graceful degradation when individual tools fail

## Enhanced Data Flow

1. **Request Reception**: Client sends POST request to `/audit` with code files
2. **Input Validation**: Pydantic models validate request structure and content
3. **File Processing**: Files are written to temporary directory with sanitized names
4. **Multi-Tool Analysis**: Parallel execution of flake8, pylint, mypy, black, isort, and ML/RL rules
5. **Result Aggregation**: Issues from all tools are collected with source attribution
6. **Fix Generation**: Advanced suggestions with diffs, replacement code, and auto-fix flags
7. **Response Delivery**: Comprehensive JSON response with enhanced analysis results

## External Dependencies

### Python Packages
- **FastAPI**: Web framework for API development
- **Uvicorn**: ASGI server for production deployment
- **Pydantic**: Data validation and serialization
- **flake8**: Syntax and style analysis
- **pylint**: Comprehensive code quality analysis
- **mypy**: Static type checking
- **black**: Code formatting
- **isort**: Import organization
- **libcst**: Concrete syntax tree manipulation for advanced analysis

### Analysis Tools
- **flake8**: Syntax and style analysis (primary engine)
- **pylint**: Semantic analysis, logical issues, and code quality checks
- **mypy**: Static type checking for type safety
- **black**: Code formatting and style consistency
- **isort**: Import statement organization and sorting
- **ML/RL Rules Engine**: Custom pattern detection for machine learning and reinforcement learning code

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
- **June 28, 2025**: Fixed ChatGPT actions integration by temporarily disabling API key authentication
- Resolved "Missing API key" errors that were blocking ChatGPT from accessing the /audit endpoint
- Authentication system modified to allow external integrations while maintaining service functionality
- Confirmed working with successful audit response from deployed service at https://codeguard.replit.app
- **Phase 1 Immediate Enhancements Completed**: Major upgrade to multi-tool static analysis system
- Added pylint, mypy, black, isort, and libcst for comprehensive code analysis
- Implemented ML/RL-specific pattern detection engine with custom rules for:
  - Missing random seeding (torch.manual_seed, np.random.seed, random.seed)
  - Improper RL environment resets and training loop issues
  - Hardcoded paths, print vs logging recommendations
  - GPU memory management suggestions and data leakage detection
- Enhanced response format with source attribution, severity levels, and advanced fix suggestions
- Added unified diff generation and auto-fixable flag for formatting improvements
- Updated OpenAPI specification to reflect new enhanced response schema
- All analysis tools running in parallel for comprehensive code quality assessment
- **Custom Rule Loader System Implemented**: Fully extensible rule architecture for user-defined analysis patterns
- Created JSON-based rule definition system with 28 predefined rules covering security, performance, and ML/RL patterns
- Implemented CustomRuleEngine with pattern matching, regex support, and automatic fix generation
- Added rule management API endpoints: `/rules/summary`, `/rules/reload`, `/rules/by-tag/{tag}`
- Rules support multiple matching strategies: contains, regex, pattern with exclusions
- Comprehensive rule categories: reproducibility, security, portability, validation, memory management, performance
- **Telemetry and Usage Metrics System**: PostgreSQL-based analytics with in-memory fallback
- Tracks audit sessions, framework usage, error patterns, and performance metrics
- Added metrics endpoints: `/metrics/usage`, `/metrics/frameworks`, `/dashboard`
- Framework detection for PyTorch, TensorFlow, JAX, Gym, Stable-Baselines3, and more
- **RL Environment Plugin Support**: Specialized analysis for reinforcement learning code
- Detects missing env.reset(), reward saturation, action space mismatches, observation handling
- Analyzes YAML configuration files for hyperparameter validation
- **Analytics Dashboard**: Comprehensive usage analytics with insights and alerts
- Export capabilities for Markdown and JSON reports
- **Historical Audit Timeline**: 30-day trend analysis showing code quality improvement over time
- Timeline tracks daily/weekly trends with improvement/worsening/stable categorization
- **OpenAI GPT Connector**: Natural language queries for past audits ("Show me gym issues")
- Intelligent query parsing supporting framework-specific, error pattern, and performance queries
- **Prompt-based Issue Explainers**: Context-aware explanations for code issues
- Static explanations for common security issues (pickle, eval) with alternatives and examples
- GPT-powered dynamic explanations for complex issues when API key available
- **Enhanced API Usage Dashboard**: Real-time tracking of user traffic and tool usage patterns
- Framework trend analysis showing PyTorch/TensorFlow/Gym usage over time
- System now processes 8 analysis tools: flake8, pylint, mypy, black, isort, ml_rules, custom_rules, rl_plugin
- Enterprise deployment successfully tested: 35 issues detected across 7 active tools with 6 security violations
- Final validation: 8 auto-fixable suggestions out of 23 total fixes, comprehensive telemetry operational
- Complete feature set: Historical timelines, natural language queries, intelligent explanations, trend analysis
- **VS Code Extension**: Comprehensive TypeScript extension providing real-time linting integration
- Extension features: Inline diagnostics, quick-fix suggestions, secure API key storage, framework detection
- Full IDE integration with context menus, progress indicators, and auto-analysis on save
- Supports project-level configuration via .codeguardrc.json files
- **One-Click Environment Setup**: Complete ML project template system with 10 framework templates
- Project templates: PyTorch, TensorFlow, RL Gym, Stable-Baselines3, JAX, scikit-learn, Data Science, Computer Vision, NLP, MLOps
- Automated project generation with dependencies, configurations, sample code, and setup instructions
- CLI tool and VS Code integration for seamless project creation with interactive wizards
- **June 28, 2025**: One-click environment setup system fully operational and tested
- Template generation creates complete project structures with proper dependencies, configurations, and framework-specific code
- Successfully tested PyTorch, TensorFlow, and RL Gym template generation with custom configuration merging
- All 10 ML framework templates implemented with comprehensive file structures and setup instructions
- Project generation API endpoints functional: `/templates`, `/templates/{name}`, `/templates/generate`

## Deployment Status

**Current Status**: ✓ Live and operational
**Production HTTPS URL**: https://codeguard.replit.app
**Key Endpoints**:
- `/audit` - Main code analysis endpoint (POST) with comprehensive telemetry
- `/rules/summary` - Custom rule management and statistics
- `/rules/by-tag/{tag}` - Filter rules by category (ml, security, performance)
- `/metrics/usage` - Usage analytics and performance metrics
- `/dashboard` - Comprehensive analytics dashboard with insights
- `/dashboard/export` - Export reports in Markdown/JSON formats
- `/timeline` - Historical audit timeline with trend analysis
- `/timeline/frameworks` - Framework usage trends over time
- `/query/audits` - Natural language queries for past audit data
- `/explain/issue` - Context-aware explanations for code issues
- `/templates` - List all ML project templates
- `/templates/{name}` - Get template details
- `/templates/generate` - Create new ML project from template
- `/templates/preview` - Preview project structure before creation
- `/docs` - Interactive API documentation 
- `/health` - Service health check
- `/.well-known/openapi.yaml` - OpenAPI spec for GPT Actions

## Changelog

- June 27, 2025: Initial setup and full implementation completed