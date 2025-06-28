"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ProjectSetupManager = void 0;
const vscode = require("vscode");
class ProjectSetupManager {
    constructor(api) {
        this.api = api;
    }
    async showProjectTemplates() {
        try {
            const templates = await this.api.getProjectTemplates();
            const items = templates.map(template => ({
                label: `$(folder) ${template.title}`,
                description: template.framework,
                detail: template.description,
                template: template
            }));
            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select a machine learning project template',
                ignoreFocusOut: true
            });
            if (selected) {
                await this.createProjectFromTemplate(selected.template);
            }
        }
        catch (error) {
            vscode.window.showErrorMessage(`Failed to load templates: ${error}`);
        }
    }
    async createProjectFromTemplate(template) {
        try {
            // Get project location
            const folderUri = await vscode.window.showOpenDialog({
                canSelectFolders: true,
                canSelectFiles: false,
                canSelectMany: false,
                openLabel: 'Select Project Location'
            });
            if (!folderUri || folderUri.length === 0) {
                return;
            }
            // Get project name
            const projectName = await vscode.window.showInputBox({
                prompt: 'Enter project name',
                value: template.name.toLowerCase().replace(/\s+/g, '_') + '_project',
                validateInput: (value) => {
                    if (!value || value.trim().length === 0) {
                        return 'Project name cannot be empty';
                    }
                    if (!/^[a-zA-Z0-9_-]+$/.test(value)) {
                        return 'Project name can only contain letters, numbers, hyphens, and underscores';
                    }
                    return null;
                }
            });
            if (!projectName) {
                return;
            }
            const projectPath = vscode.Uri.joinPath(folderUri[0], projectName).fsPath;
            // Show preview
            const preview = await this.api.previewProject(template.name);
            const createProject = await this.showProjectPreview(preview, projectPath);
            if (!createProject) {
                return;
            }
            // Create project with progress
            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Creating ML Project",
                cancellable: false
            }, async (progress) => {
                progress.report({ increment: 0, message: `Setting up ${template.title}...` });
                const result = await this.api.generateProject(template.name, projectPath);
                progress.report({ increment: 100, message: "Complete!" });
                // Show success message with actions
                const openProject = await vscode.window.showInformationMessage(`Project "${projectName}" created successfully!`, 'Open Project', 'Show Setup Guide');
                if (openProject === 'Open Project') {
                    const projectUri = vscode.Uri.file(projectPath);
                    await vscode.commands.executeCommand('vscode.openFolder', projectUri);
                }
                else if (openProject === 'Show Setup Guide') {
                    await this.showSetupGuide(result);
                }
            });
        }
        catch (error) {
            vscode.window.showErrorMessage(`Failed to create project: ${error}`);
        }
    }
    async showProjectPreview(preview, projectPath) {
        const panel = vscode.window.createWebviewPanel('projectPreview', 'Project Preview', vscode.ViewColumn.One, {
            enableScripts: true,
            localResourceRoots: []
        });
        panel.webview.html = this.getPreviewHtml(preview, projectPath);
        return new Promise((resolve) => {
            panel.webview.onDidReceiveMessage(message => {
                switch (message.command) {
                    case 'create':
                        panel.dispose();
                        resolve(true);
                        return;
                    case 'cancel':
                        panel.dispose();
                        resolve(false);
                        return;
                }
            });
            panel.onDidDispose(() => {
                resolve(false);
            });
        });
    }
    getPreviewHtml(preview, projectPath) {
        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Project Preview</title>
                <style>
                    body {
                        font-family: var(--vscode-font-family);
                        color: var(--vscode-foreground);
                        background-color: var(--vscode-editor-background);
                        padding: 20px;
                        line-height: 1.6;
                    }
                    .header {
                        border-bottom: 1px solid var(--vscode-panel-border);
                        padding-bottom: 15px;
                        margin-bottom: 20px;
                    }
                    .section {
                        margin-bottom: 25px;
                    }
                    .section h3 {
                        color: var(--vscode-textLink-foreground);
                        margin-bottom: 10px;
                    }
                    .file-list, .dir-list {
                        background-color: var(--vscode-editor-inactiveSelectionBackground);
                        padding: 10px;
                        border-radius: 4px;
                        margin-bottom: 10px;
                    }
                    .file-item, .dir-item {
                        padding: 2px 0;
                        font-family: var(--vscode-editor-font-family);
                    }
                    .buttons {
                        display: flex;
                        gap: 10px;
                        margin-top: 30px;
                        padding-top: 20px;
                        border-top: 1px solid var(--vscode-panel-border);
                    }
                    button {
                        padding: 10px 20px;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 14px;
                    }
                    .create-btn {
                        background-color: var(--vscode-button-background);
                        color: var(--vscode-button-foreground);
                    }
                    .cancel-btn {
                        background-color: var(--vscode-button-secondaryBackground);
                        color: var(--vscode-button-secondaryForeground);
                    }
                    .path {
                        font-family: var(--vscode-editor-font-family);
                        background-color: var(--vscode-input-background);
                        padding: 8px;
                        border-radius: 4px;
                        margin-top: 10px;
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>🚀 ${preview.template_name} Project</h2>
                    <p><strong>Framework:</strong> ${preview.framework}</p>
                    <p><strong>Location:</strong></p>
                    <div class="path">${projectPath}</div>
                </div>
                
                <div class="section">
                    <h3>📊 Project Summary</h3>
                    <ul>
                        <li>Files to create: <strong>${preview.files_to_create.length}</strong></li>
                        <li>Directories: <strong>${preview.directories_to_create.length}</strong></li>
                        <li>Dependencies: <strong>${preview.dependencies_count}</strong></li>
                    </ul>
                </div>
                
                <div class="section">
                    <h3>📄 Files</h3>
                    <div class="file-list">
                        ${preview.files_to_create.map(file => `<div class="file-item">📄 ${file}</div>`).join('')}
                    </div>
                </div>
                
                <div class="section">
                    <h3>📁 Directories</h3>
                    <div class="dir-list">
                        ${preview.directories_to_create.map(dir => `<div class="dir-item">📁 ${dir}/</div>`).join('')}
                    </div>
                </div>
                
                <div class="section">
                    <h3>🛠️ Setup Commands</h3>
                    <div class="file-list">
                        ${preview.setup_commands.map((cmd, i) => `<div class="file-item">${i + 1}. ${cmd}</div>`).join('')}
                    </div>
                </div>
                
                <div class="buttons">
                    <button class="create-btn" onclick="createProject()">Create Project</button>
                    <button class="cancel-btn" onclick="cancel()">Cancel</button>
                </div>
                
                <script>
                    const vscode = acquireVsCodeApi();
                    
                    function createProject() {
                        vscode.postMessage({
                            command: 'create'
                        });
                    }
                    
                    function cancel() {
                        vscode.postMessage({
                            command: 'cancel'
                        });
                    }
                </script>
            </body>
            </html>
        `;
    }
    async showSetupGuide(projectResult) {
        const panel = vscode.window.createWebviewPanel('setupGuide', 'Setup Guide', vscode.ViewColumn.One, {
            enableScripts: true,
            localResourceRoots: []
        });
        panel.webview.html = this.getSetupGuideHtml(projectResult);
    }
    getSetupGuideHtml(project) {
        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Setup Guide</title>
                <style>
                    body {
                        font-family: var(--vscode-font-family);
                        color: var(--vscode-foreground);
                        background-color: var(--vscode-editor-background);
                        padding: 20px;
                        line-height: 1.6;
                    }
                    .step {
                        background-color: var(--vscode-editor-inactiveSelectionBackground);
                        padding: 15px;
                        border-radius: 6px;
                        margin-bottom: 15px;
                        border-left: 4px solid var(--vscode-textLink-foreground);
                    }
                    .command {
                        background-color: var(--vscode-terminal-background);
                        color: var(--vscode-terminal-foreground);
                        padding: 10px;
                        border-radius: 4px;
                        font-family: var(--vscode-editor-font-family);
                        margin: 8px 0;
                        overflow-x: auto;
                    }
                    .success {
                        color: var(--vscode-testing-iconPassed);
                        font-weight: bold;
                    }
                </style>
            </head>
            <body>
                <h2>🎉 Project Setup Complete!</h2>
                
                <div class="step">
                    <h3>1. Navigate to Project</h3>
                    <div class="command">cd ${project.project_path}</div>
                </div>
                
                <div class="step">
                    <h3>2. Create Virtual Environment</h3>
                    <div class="command">python -m venv venv</div>
                </div>
                
                <div class="step">
                    <h3>3. Activate Virtual Environment</h3>
                    <div class="command">
                        # Linux/Mac:<br>
                        source venv/bin/activate<br><br>
                        # Windows:<br>
                        venv\\Scripts\\activate
                    </div>
                </div>
                
                <div class="step">
                    <h3>4. Install Dependencies</h3>
                    <div class="command">pip install -r requirements.txt</div>
                </div>
                
                ${project.setup_commands ? project.setup_commands.map((cmd, i) => `
                <div class="step">
                    <h3>${5 + i}. Setup Command</h3>
                    <div class="command">${cmd}</div>
                </div>
                `).join('') : ''}
                
                <div class="step">
                    <h3>🚀 Run Your Project</h3>
                    <div class="command">python main.py</div>
                </div>
                
                <p class="success">✅ Your ${project.framework} project is ready to go!</p>
            </body>
            </html>
        `;
    }
}
exports.ProjectSetupManager = ProjectSetupManager;
//# sourceMappingURL=project_setup.js.map