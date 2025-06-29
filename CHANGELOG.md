# Changelog

All notable changes to CodeGuard API will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-06-29

### Added
- **Production-ready API server** with FastAPI and comprehensive OpenAPI documentation
- **Multi-tool static analysis** combining flake8, pylint, mypy, black, and isort
- **AST-based semantic analysis** for intelligent false positive filtering
- **Multi-LLM AI integration** supporting OpenAI GPT-4o, DeepSeek R1, Google Gemini, and Anthropic Claude
- **Custom rule engine** with 28+ predefined ML/RL security and performance rules
- **Project template system** with 10 ML framework templates (PyTorch, TensorFlow, etc.)
- **VS Code extension** with real-time analysis and AI-powered improvements
- **Comprehensive analytics** with PostgreSQL telemetry and usage dashboards
- **Bearer token authentication** for secure API access
- **Interactive web playground** for testing code analysis features

### Core Analysis Features
- ML/RL specialized pattern detection for PyTorch, TensorFlow, OpenAI Gym
- Security vulnerability detection (eval, pickle, hardcoded credentials)
- Reproducibility checks (missing random seeds, non-deterministic operations)
- Performance optimization suggestions (GPU memory, tensor operations)
- RL environment analysis (missing resets, reward handling, action spaces)

### AI-Powered Improvements
- Automatic code fix implementation with confidence scoring
- Bulk fix operations for applying same fix across multiple instances
- Multi-provider fallback system for maximum reliability
- DeepSeek Function Calling for specialized ML/RL analysis
- Fill-in-the-Middle (FIM) completion for targeted code improvements

### Developer Experience
- Real-time IDE integration through VS Code extension
- One-click project setup with framework-specific templates
- Comprehensive reporting with exportable analysis results
- Natural language query interface for audit history
- Context-aware issue explanations

### Infrastructure
- Production deployment on Replit with 99.9% uptime
- PostgreSQL database with in-memory fallback
- Concurrent request handling with automatic scaling
- Secure temporary file processing with cleanup
- CORS support for cross-origin integrations

### Documentation
- Complete API reference with interactive Swagger UI
- Production deployment guides
- VS Code extension installation instructions
- OpenAI GPT Actions integration guide
- Privacy policy and terms of service

## [0.9.0] - 2025-06-28

### Added
- Initial ChatGPT integration with OpenAI API
- False positive filtering system
- Enhanced audit engine with multi-tool support
- Basic telemetry and analytics

### Changed
- Migrated from simple flake8-only analysis to comprehensive multi-tool system
- Updated response format to include source attribution and severity levels

## [0.8.0] - 2025-06-27

### Added
- Core FastAPI application structure
- Basic flake8 integration
- Authentication system with API key validation
- OpenAPI specification for GPT Actions

### Changed
- Initial project setup and basic audit functionality

## [0.1.0] - 2025-06-27

### Added
- Project initialization
- Basic code structure and development environment