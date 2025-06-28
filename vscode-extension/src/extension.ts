import * as vscode from 'vscode';
import { CodeGuardAPI } from './api';
import { DiagnosticsManager } from './diagnostics';
import { ConfigManager } from './config';
import { ProjectSetupManager } from './project_setup';

let diagnosticsManager: DiagnosticsManager;
let api: CodeGuardAPI;
let configManager: ConfigManager;
let projectSetupManager: ProjectSetupManager;

export function activate(context: vscode.ExtensionContext) {
    console.log('CodeGuard extension is now active!');
    
    // Initialize components
    configManager = new ConfigManager(context);
    api = new CodeGuardAPI(configManager);
    diagnosticsManager = new DiagnosticsManager();
    projectSetupManager = new ProjectSetupManager(api);
    
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
    
    const setupProjectCommand = vscode.commands.registerCommand('codeguard.setupProject', async () => {
        await projectSetupManager.showProjectTemplates();
    });
    
    const applySingleFixCommand = vscode.commands.registerCommand('codeguard.applySingleFix', async (fix) => {
        await applySingleFix(fix);
    });
    
    const improveWithChatGPTCommand = vscode.commands.registerCommand('codeguard.improveWithChatGPT', async () => {
        await improveCurrentFileWithChatGPT();
    });
    
    const showFixMenuCommand = vscode.commands.registerCommand('codeguard.showFixMenu', async () => {
        await showFixSelectionMenu();
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
        setupProjectCommand,
        applySingleFixCommand,
        improveWithChatGPTCommand,
        showFixMenuCommand,
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

async function applySingleFix(fix: any) {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showWarningMessage('No active file to apply fix');
        return;
    }

    try {
        if (fix.replacement_code) {
            // Apply the fix directly
            const edit = new vscode.WorkspaceEdit();
            const range = new vscode.Range(
                fix.line - 1, 0,
                fix.line - 1, activeEditor.document.lineAt(fix.line - 1).text.length
            );
            edit.replace(activeEditor.document.uri, range, fix.replacement_code);
            await vscode.workspace.applyEdit(edit);
            
            vscode.window.showInformationMessage(`Applied fix: ${fix.description}`);
        } else {
            // Show fix details for manual application
            const apply = await vscode.window.showInformationMessage(
                `Fix suggestion: ${fix.description}\n\nLine ${fix.line}: ${fix.suggestion}`,
                'Apply Manual Fix', 'Dismiss'
            );
            
            if (apply === 'Apply Manual Fix') {
                // Navigate to the line
                const position = new vscode.Position(fix.line - 1, 0);
                activeEditor.selection = new vscode.Selection(position, position);
                activeEditor.revealRange(new vscode.Range(position, position));
            }
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to apply fix: ${error}`);
    }
}

async function improveCurrentFileWithChatGPT() {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showWarningMessage('No active Python file to improve');
        return;
    }

    if (activeEditor.document.languageId !== 'python') {
        vscode.window.showWarningMessage('ChatGPT improvement only supports Python files');
        return;
    }

    try {
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "ChatGPT Code Improvement",
            cancellable: false
        }, async (progress) => {
            progress.report({ increment: 0, message: "Analyzing with CodeGuard..." });

            const filename = activeEditor.document.fileName.split('/').pop() || 'untitled.py';
            const content = activeEditor.document.getText();

            // First get CodeGuard analysis
            const auditResult = await api.auditCode([{ filename, content }]);
            
            progress.report({ increment: 30, message: "Requesting ChatGPT improvements..." });

            // Then get ChatGPT improvements
            const improvement = await api.improveCode(content, filename, auditResult.issues, auditResult.fixes);
            
            progress.report({ increment: 80, message: "Applying improvements..." });

            // Show the improved code in a new editor
            const doc = await vscode.workspace.openTextDocument({
                content: improvement.improved_code,
                language: 'python'
            });
            
            await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
            
            progress.report({ increment: 100, message: "Complete!" });

            // Show improvement summary
            const applyChanges = await vscode.window.showInformationMessage(
                `ChatGPT Improvements Applied:\n${improvement.improvement_summary}\n\nConfidence: ${Math.round(improvement.confidence_score * 100)}%`,
                'Replace Original', 'Keep Both', 'Dismiss'
            );

            if (applyChanges === 'Replace Original') {
                // Replace the original file content
                const edit = new vscode.WorkspaceEdit();
                const fullRange = new vscode.Range(
                    0, 0,
                    activeEditor.document.lineCount, 0
                );
                edit.replace(activeEditor.document.uri, fullRange, improvement.improved_code);
                await vscode.workspace.applyEdit(edit);
                vscode.window.showInformationMessage('Original file updated with ChatGPT improvements');
            }
        });

    } catch (error) {
        vscode.window.showErrorMessage(`ChatGPT improvement failed: ${error}`);
        console.error('ChatGPT improvement error:', error);
    }
}

async function showFixSelectionMenu() {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showWarningMessage('No active file');
        return;
    }

    // Get current diagnostics for the file
    const diagnostics = diagnosticsManager.getDiagnosticsForFile(activeEditor.document.uri);
    
    if (!diagnostics || diagnostics.length === 0) {
        vscode.window.showInformationMessage('No CodeGuard issues found. Run audit first.');
        return;
    }

    // Create quick pick items for each fix
    const fixItems = diagnostics.map((diagnostic, index) => ({
        label: `Line ${diagnostic.range.start.line + 1}: ${diagnostic.source}`,
        description: diagnostic.message,
        detail: diagnostic.code ? `Code: ${diagnostic.code}` : '',
        diagnostic,
        index
    }));

    const selected = await vscode.window.showQuickPick(fixItems, {
        placeHolder: 'Select an issue to fix with ChatGPT',
        matchOnDescription: true,
        matchOnDetail: true
    });

    if (selected) {
        // Apply ChatGPT fix for this specific issue
        await applySpecificChatGPTFix(selected.diagnostic);
    }
}

async function applySpecificChatGPTFix(diagnostic: vscode.Diagnostic) {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) return;

    try {
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Applying ChatGPT Fix",
            cancellable: false
        }, async (progress) => {
            progress.report({ increment: 0, message: "Analyzing specific issue..." });

            const filename = activeEditor.document.fileName.split('/').pop() || 'untitled.py';
            const content = activeEditor.document.getText();
            const line = diagnostic.range.start.line + 1;

            // Create a focused issue for ChatGPT
            const focusedIssue = {
                type: diagnostic.source || 'unknown',
                severity: diagnostic.severity === vscode.DiagnosticSeverity.Error ? 'error' : 'warning',
                line: line,
                column: diagnostic.range.start.character + 1,
                description: diagnostic.message,
                source: diagnostic.source,
                code: diagnostic.code
            };

            progress.report({ increment: 50, message: "Getting ChatGPT suggestion..." });

            // Get targeted improvement from ChatGPT
            const improvement = await api.improveCode(content, filename, [focusedIssue], []);
            
            progress.report({ increment: 100, message: "Complete!" });

            // Show the specific fix
            const apply = await vscode.window.showInformationMessage(
                `ChatGPT Fix for Line ${line}:\n${improvement.improvement_summary}`,
                'Preview Changes', 'Apply Fix', 'Dismiss'
            );

            if (apply === 'Preview Changes') {
                // Show diff view
                const doc = await vscode.workspace.openTextDocument({
                    content: improvement.improved_code,
                    language: 'python'
                });
                await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
                
            } else if (apply === 'Apply Fix') {
                // Apply the fix
                const edit = new vscode.WorkspaceEdit();
                const fullRange = new vscode.Range(
                    0, 0,
                    activeEditor.document.lineCount, 0
                );
                edit.replace(activeEditor.document.uri, fullRange, improvement.improved_code);
                await vscode.workspace.applyEdit(edit);
                
                // Clear the specific diagnostic
                diagnosticsManager.clearSpecificDiagnostic(activeEditor.document.uri, diagnostic);
                
                vscode.window.showInformationMessage('ChatGPT fix applied successfully');
            }
        });

    } catch (error) {
        vscode.window.showErrorMessage(`Failed to apply ChatGPT fix: ${error}`);
    }
}

export function deactivate() {
    if (diagnosticsManager) {
        diagnosticsManager.dispose();
    }
}