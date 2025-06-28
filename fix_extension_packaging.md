# Fix VS Code Extension Packaging Issue

## Problem
VSIX packaging fails due to duplicate LICENSE files with same case-insensitive path.

## Solution Steps

1. **Clean your local extension directory:**
```bash
cd ~/codeGuard/PersonalTracker/vscode-extension
find . -name "LICENSE*" -delete
find . -name "*.txt" -delete
```

2. **Copy the single LICENSE file:**
```bash
# Copy only the LICENSE file from Replit (no .txt extension)
# Ensure only one LICENSE file exists
```

3. **Check for duplicate files:**
```bash
find . -name "*" -type f | sort | uniq -d
```

4. **Clean node_modules and rebuild:**
```bash
rm -rf node_modules
npm install
npm run compile
```

5. **Package without sudo:**
```bash
vsce package
```

## Alternative: Use the working v0.1.0 extension
Since your v0.1.0 extension works perfectly, you can continue using it. The ChatGPT features are available through the backend API regardless of extension version.

## Quick Fix Commands:
```bash
cd ~/codeGuard/PersonalTracker/vscode-extension
find . -name "LICENSE*" -delete
echo 'MIT License...' > LICENSE
npm run compile
vsce package
```