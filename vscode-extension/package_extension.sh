#!/bin/bash

# Simple VS Code extension packaging script
echo "Packaging CodeGuard VS Code Extension v0.2.0..."

# Create package directory
mkdir -p package
cp -r out package/
cp package.json package/
cp LICENSE package/
cp CHANGELOG.md package/
cp README.md package/
cp -r src package/

# Create basic VSIX structure
cd package
zip -r ../codeguard-0.2.0.vsix . -x "node_modules/*" "*.git*" "*.DS_Store*" "test-files/*"

cd ..
echo "Extension packaged: codeguard-0.2.0.vsix"
ls -la *.vsix