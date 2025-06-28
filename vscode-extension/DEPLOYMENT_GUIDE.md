# CodeGuard VS Code Extension - Deployment Guide

## Development Setup

### Prerequisites
- Node.js 16+ 
- npm or yarn
- VS Code 1.80+
- TypeScript 4.9+

### Local Development

1. **Clone and Setup**
```bash
cd vscode-extension
npm install
npm run compile
```

2. **Development Testing**
```bash
# Start watch mode for auto-compilation
npm run watch

# In VS Code: Press F5 to launch Extension Development Host
# This opens a new VS Code window with the extension loaded
```

3. **Manual Testing**
```bash
# Create symlink for local testing
./install-dev.sh
```

## Building for Distribution

### Package Extension
```bash
# Install vsce if not present
npm install -g vsce

# Package extension
npm run package
# or
vsce package

# Output: codeguard-0.1.0.vsix
```

### Local Installation
```bash
# Install from package file
code --install-extension codeguard-0.1.0.vsix

# Or through VS Code UI:
# Extensions → "..." → Install from VSIX
```

## Publishing

### VS Code Marketplace

1. **Setup Publisher Account**
```bash
# Create publisher (one-time setup)
vsce create-publisher your-publisher-name
```

2. **Get Personal Access Token**
- Visit: https://dev.azure.com/
- Create PAT with Marketplace permissions
- Store securely

3. **Login and Publish**
```bash
vsce login your-publisher-name
vsce publish
```

### Private Distribution

For enterprise/private use:

1. **Share VSIX File**
- Build: `vsce package`
- Distribute: `codeguard-0.1.0.vsix`
- Install: `code --install-extension codeguard-0.1.0.vsix`

2. **Internal Registry**
- Host on internal server
- Use `--install-extension` with URL

## Configuration

### Required Settings
Users must configure:
- `codeguard.apiKey`: API key for authentication
- `codeguard.serverUrl`: Backend URL (default: https://codeguard.replit.app)

### Optional Settings
- `codeguard.auditOnSave`: Auto-analyze on save (default: true)
- `codeguard.analysisLevel`: basic/standard/strict (default: standard)
- `codeguard.ignoreRules`: Rules to skip (default: [])

## Testing

### Unit Tests
```bash
npm test
```

### Integration Testing
1. Open Extension Development Host
2. Create Python file with test code
3. Verify diagnostics appear
4. Test commands work
5. Check quick-fixes function

### Sample Test File
Use `test-files/sample_ml_code.py` for comprehensive testing

## Troubleshooting

### Common Issues

**Extension Not Loading**
- Check VS Code version compatibility
- Verify TypeScript compilation: `npm run compile`
- Check Developer Tools: Help → Toggle Developer Tools

**API Connection Fails**
- Verify server URL in settings
- Check API key configuration
- Test with: `curl -H "Authorization: Bearer YOUR_KEY" SERVER_URL/health`

**Diagnostics Not Showing**
- Ensure Python file is active
- Check if auto-analysis is enabled
- Manually trigger: Ctrl+Shift+P → "CodeGuard: Run Audit"

**Build Failures**
- Update dependencies: `npm update`
- Clear node_modules: `rm -rf node_modules && npm install`
- Check TypeScript version compatibility

## Monitoring

### Usage Analytics
- VS Code provides built-in extension analytics
- Track activation events and command usage
- Monitor error rates through telemetry

### Performance
- Extension loads on Python file open
- API calls average 2-5 seconds
- Memory usage typically <50MB

## Updates

### Version Management
- Update `package.json` version
- Update `CHANGELOG.md`
- Tag release: `git tag v0.1.1`
- Publish: `vsce publish patch`

### Auto-Updates
- Users receive automatic updates from marketplace
- Enterprise: Update VSIX file distribution