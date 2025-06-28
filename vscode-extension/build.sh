#!/bin/bash

# CodeGuard VS Code Extension Build Script
echo "Building CodeGuard VS Code Extension..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "npm is not installed. Please install npm first."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
npm install

# Install vsce if not present
if ! command -v vsce &> /dev/null; then
    echo "Installing vsce (VS Code Extension CLI)..."
    npm install -g vsce
fi

# Compile TypeScript
echo "Compiling TypeScript..."
npm run compile

if [ $? -ne 0 ]; then
    echo "TypeScript compilation failed"
    exit 1
fi

# Package extension
echo "Packaging extension..."
vsce package

if [ $? -eq 0 ]; then
    echo "Extension packaged successfully!"
    echo "Generated: codeguard-0.1.0.vsix"
    echo ""
    echo "To install locally:"
    echo "   code --install-extension codeguard-0.1.0.vsix"
    echo ""
    echo "To publish to marketplace:"
    echo "   vsce publish"
else
    echo "Extension packaging failed"
    exit 1
fi