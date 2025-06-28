import * as vscode from 'vscode';
import { CodeGuardAPI } from './api';
import { DiagnosticsManager } from './diagnostics';
import { ConfigManager } from './config';

let diagnosticsManager: DiagnosticsManager;
let api: CodeGuardAPI;
let configManager: ConfigManager;

export function activate(context: vscode.ExtensionContext) {
    console.log('CodeGuard extension is now active!');
    
    // Initialize components
    configManager = new ConfigManager(context);
    api = new CodeGuardAPI(configManager);
    diagnosticsManager = new DiagnosticsManager();
    
    // Register commands
    const runAuditCommand = vscode.commands.registerCommand('codeguard.runAudit', async () => {
        await runAudit();
    });
    
    const clearDiagnosticsCommand = vscode.commands.registerCommand('codeguard.clearDiagnostics', () => {
        diagnosticsManager.clearAll();
        vscode.window.showInformationMessage('CodeGuard diagnostics cleared');
    });
    
    const generateReportCommand = vscode.commands.registerCommand('codeguard.generateReport', async () => {
        await generateReport();
    });
    
    // Register event listeners
    const onSaveListener = vscode.workspace.onDidSaveTextDocument(async (document) => {
        if (document.languageId === 'python' && configManager.getAuditOnSave()) {
            await runAuditForDocument(document);
        }
    });
    
    // Add to context subscriptions
    context.subscriptions.push(
        runAuditCommand,
        clearDiagnosticsCommand,
        generateReportCommand,
        onSaveListener,
        diagnosticsManager.diagnosticCollection
    );
    
    // Show welcome message
    vscode.window.showInformationMessage('CodeGuard ML/RL Auditor is ready!');
}

async function runAudit() {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showWarningMessage('No active Python file to audit');
        return;
    }
    
    if (activeEditor.document.languageId !== 'python') {
        vscode.window.showWarningMessage('CodeGuard only supports Python files');
        return;
    }
    
    await runAuditForDocument(activeEditor.document);
}

async function runAuditForDocument(document: vscode.TextDocument) {
    if (!await configManager.isConfigured()) {
        const configure = await vscode.window.showErrorMessage(
            'CodeGuard API key not configured. Would you like to configure it now?',
            'Configure'
        );
        if (configure === 'Configure') {
            await configManager.promptForApiKey();
        }
        return;
    }
    
    try {
        // Show progress
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "CodeGuard Analysis",
            cancellable: false
        }, async (progress) => {
            progress.report({ increment: 0, message: "Analyzing code..." });
            
            const files = [{
                filename: document.fileName.split('/').pop() || 'untitled.py',
                content: document.getText()
            }];
            
            progress.report({ increment: 50, message: "Sending to CodeGuard API..." });
            
            const result = await api.auditCode(files);
            
            progress.report({ increment: 80, message: "Processing results..." });
            
            // Clear previous diagnostics for this file
            diagnosticsManager.clearForFile(document.uri);
            
            // Add new diagnostics
            diagnosticsManager.addDiagnostics(document.uri, result.issues, result.fixes);
            
            progress.report({ increment: 100, message: "Complete!" });
            
            // Show summary
            const issueCount = result.issues.length;
            const fixCount = result.fixes.filter(f => f.auto_fixable).length;
            vscode.window.showInformationMessage(
                `CodeGuard: Found ${issueCount} issues, ${fixCount} auto-fixable`
            );
        });
        
    } catch (error) {
        vscode.window.showErrorMessage(`CodeGuard analysis failed: ${error}`);
        console.error('CodeGuard error:', error);
    }
}

async function generateReport() {
    try {
        const result = await vscode.window.showQuickPick([
            { label: 'Markdown Report', value: 'markdown' },
            { label: 'JSON Report', value: 'json' }
        ], {
            placeHolder: 'Select report format'
        });
        
        if (!result) return;
        
        const reportData = await api.generateReport(7, result.value);
        
        // Create new document with report
        const doc = await vscode.workspace.openTextDocument({
            content: reportData.report,
            language: result.value === 'markdown' ? 'markdown' : 'json'
        });
        
        await vscode.window.showTextDocument(doc);
        
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to generate report: ${error}`);
    }
}

export function deactivate() {
    if (diagnosticsManager) {
        diagnosticsManager.dispose();
    }
}