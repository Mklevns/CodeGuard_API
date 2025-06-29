# Contributing to CodeGuard

We welcome contributions to CodeGuard! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites
- Python 3.11+
- PostgreSQL (optional, for telemetry features)
- Node.js 16+ (for VS Code extension development)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd codeguard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start development server**
   ```bash
   python main.py
   ```

4. **Run tests**
   ```bash
   python -m pytest tests/
   ```

## Project Structure

```
CodeGuard/
├── main.py                    # API server entry point
├── models.py                  # Pydantic data models
├── audit.py                   # Core audit functionality
├── auth.py                    # Authentication system
├── enhanced_audit.py          # Multi-tool analysis engine
├── false_positive_filter.py   # AST-based filtering
├── semantic_analyzer.py       # Semantic code analysis
├── rule_engine.py            # Custom rule processing
├── rule_loader.py            # Rule management system
├── chatgpt_integration.py    # AI-powered improvements
├── multi_ai_integration.py   # Multi-LLM support
├── telemetry.py              # Usage analytics
├── dashboard.py              # Analytics dashboard
├── project_templates.py      # ML project templates
├── static/                   # Web playground assets
├── rules/                    # JSON rule definitions
├── vscode-extension/         # VS Code extension
├── tests/                    # Test suites
├── docs/                     # Documentation
└── scripts/                  # Deployment scripts
```

## Code Style Guidelines

### Python Code
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Maximum line length: 88 characters (Black formatter)
- Use docstrings for all public functions and classes

### API Design
- RESTful endpoint design
- Consistent error handling
- Comprehensive request/response validation
- OpenAPI 3.1.0 specification compliance

### Testing
- Unit tests for core functionality
- Integration tests for API endpoints
- Test coverage minimum: 80%
- Use pytest framework

## Contributing Process

### 1. Issue Creation
- Search existing issues before creating new ones
- Use issue templates for bug reports and feature requests
- Provide clear descriptions and reproduction steps

### 2. Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Update documentation if needed
7. Commit with clear messages: `git commit -m "Add feature X"`

### 3. Pull Request Process
1. Push your branch: `git push origin feature/your-feature-name`
2. Create a pull request with:
   - Clear title and description
   - Link to related issues
   - Screenshots for UI changes
   - Test results
3. Address code review feedback
4. Ensure CI passes

## Development Areas

### Core Analysis Engine
- **Static Analysis Tools**: Enhance flake8, pylint, mypy integration
- **Custom Rules**: Add new ML/RL pattern detection rules
- **Performance**: Optimize analysis speed and memory usage

### AI Integration
- **New Providers**: Add support for additional LLM providers
- **Improvement Quality**: Enhance code improvement suggestions
- **Error Handling**: Better fallback mechanisms

### VS Code Extension
- **Features**: Add new IDE integration features
- **UI/UX**: Improve user experience
- **Performance**: Optimize extension responsiveness

### Documentation
- **API Docs**: Improve endpoint documentation
- **Tutorials**: Create user guides and tutorials
- **Examples**: Add more code examples

## Testing Guidelines

### Unit Tests
```python
def test_audit_functionality():
    """Test core audit functionality."""
    request = AuditRequest(files=[...])
    response = analyze_code(request)
    assert response.issues is not None
```

### Integration Tests
```python
def test_api_endpoint():
    """Test API endpoint integration."""
    response = client.post("/audit", json=test_data)
    assert response.status_code == 200
```

### Test Data
- Use realistic ML/RL code samples
- Cover edge cases and error conditions
- Test with different framework combinations

## Documentation Standards

### Code Documentation
- Docstrings for all public APIs
- Inline comments for complex logic
- Type hints for better IDE support

### API Documentation
- OpenAPI specification maintenance
- Example requests/responses
- Error code documentation

### User Documentation
- Clear installation instructions
- Usage examples
- Troubleshooting guides

## Release Process

### Version Numbering
- Semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR: Breaking API changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### Release Checklist
1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create release tag
6. Deploy to production
7. Announce release

## Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers get started
- Report inappropriate behavior

### Communication Channels
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Discord: Real-time community chat

## Getting Help

### Documentation
- [README.md](README.md): Project overview
- [API Documentation](docs/api/): Complete API reference
- [Deployment Guide](docs/guides/DEPLOYMENT.md): Production setup

### Community Support
- GitHub Issues: Technical problems
- Discord Community: General questions
- Stack Overflow: Tag questions with `codeguard`

## Recognition

Contributors are recognized in:
- CHANGELOG.md for each release
- README.md contributors section
- GitHub contributor graphs

Thank you for contributing to CodeGuard! 🚀