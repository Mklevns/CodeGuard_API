# CodeGuard VS Code Extension

Enterprise-grade static analysis for ML/RL Python code with integrated framework detection, custom rules, and inline fix suggestions.

## Features

- **Real-time Analysis**: Automatic code analysis using CodeGuard's 8-tool engine
- **Framework Detection**: Intelligent detection of PyTorch, TensorFlow, Gym, and more
- **Inline Diagnostics**: Issues highlighted directly in your code with quick-fix suggestions
- **Custom Rules**: Support for project-specific rule configurations
- **Security Scanning**: Detects security vulnerabilities like unsafe pickle usage
- **Auto-fixes**: One-click fixes for formatting and common issues

## Installation

1. Install the extension from the VS Code Marketplace
2. Configure your CodeGuard API key: `Ctrl+Shift+P` → "CodeGuard: Configure"
3. Start coding! The extension will automatically analyze your Python files

## Configuration

### Required Settings

- **API Key**: Your CodeGuard API key (stored securely)
- **Server URL**: CodeGuard backend URL (default: https://codeguard.replit.app)

### Optional Settings

- **Audit on Save**: Automatically analyze files when saved (default: true)
- **Analysis Level**: basic, standard, or strict (default: standard)
- **Ignore Rules**: List of rule IDs to skip

### Example Settings

```json
{
  "codeguard.serverUrl": "https://codeguard.replit.app",
  "codeguard.auditOnSave": true,
  "codeguard.analysisLevel": "strict",
  "codeguard.ignoreRules": ["F401", "W293"]
}
```

## Commands

- **CodeGuard: Run Audit** - Manually trigger analysis
- **CodeGuard: Clear Diagnostics** - Clear all issue highlights
- **CodeGuard: Generate Report** - Export analysis report

## Project Configuration

Create a `.codeguardrc.json` file in your project root for project-specific settings:

```json
{
  "analysisLevel": "strict",
  "ignoreRules": ["custom_rule_1"],
  "framework": "pytorch"
}
```

## Supported Frameworks

- PyTorch
- TensorFlow / Keras
- JAX
- OpenAI Gym
- Stable-Baselines3
- Scikit-learn
- Pandas / NumPy

## Issue Types

The extension detects various issue categories:

- **Syntax Errors**: Python syntax issues
- **Style Issues**: PEP 8 compliance
- **Type Issues**: Type safety problems
- **Security Issues**: Unsafe code patterns
- **ML/RL Issues**: Framework-specific problems
- **Performance Issues**: Optimization opportunities

## Quick Fixes

Many issues can be auto-fixed with a single click:

- Code formatting (Black)
- Import sorting (isort)
- Simple syntax fixes
- Common ML/RL patterns

## Support

For issues and feature requests, visit: https://github.com/codeguard-ai/vscode-extension

## License

MIT License - see LICENSE file for details