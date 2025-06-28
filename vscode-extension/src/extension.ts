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
    
    const bulkFixCommand = vscode.commands.registerCommand('codeguard.bulkFixByType', async () => {
        await showBulkFixMenu();
    });
    
    const generateImprovementReportCommand = vscode.commands.registerCommand('codeguard.generateImprovementReport', async () => {
        await generateComprehensiveReport();
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
        bulkFixCommand,
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

async function generateComprehensiveReport() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active Python file');
        return;
    }

    if (!editor.document.fileName.endsWith('.py')) {
        vscode.window.showErrorMessage('CodeGuard only supports Python files');
        return;
    }

    try {
        // Show format selection
        const format = await vscode.window.showQuickPick(
            ['Markdown', 'HTML', 'JSON'],
            { placeHolder: 'Select report format' }
        );
        
        if (!format) return;

        // Show AI inclusion option
        const includeAi = await vscode.window.showQuickPick(
            ['Yes', 'No'],
            { placeHolder: 'Include AI improvement suggestions?' }
        );
        
        if (!includeAi) return;

        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Generating comprehensive improvement report...",
            cancellable: false
        }, async (progress) => {
            progress.report({ increment: 20, message: "Analyzing code..." });
            
            const files = [{
                filename: editor.document.fileName.split('/').pop() || 'unknown.py',
                content: editor.document.getText()
            }];

            progress.report({ increment: 40, message: "Running analysis tools..." });
            
            const reportData = await api.generateImprovementReport(
                files, 
                format.toLowerCase(),
                includeAi === 'Yes'
            );

            progress.report({ increment: 80, message: "Formatting report..." });

            // Create and show the report
            const reportDoc = await vscode.workspace.openTextDocument({
                content: typeof reportData.report === 'string' ? reportData.report : JSON.stringify(reportData.report, null, 2),
                language: format.toLowerCase() === 'markdown' ? 'markdown' : 
                         format.toLowerCase() === 'html' ? 'html' : 'json'
            });

            await vscode.window.showTextDocument(reportDoc);
            
            progress.report({ increment: 100, message: "Report generated!" });

            // Show summary
            const summary = `Report Generated:
• ${reportData.total_files} files analyzed
• ${reportData.total_issues} issues found
• ${reportData.severity_breakdown?.error || 0} errors
• ${reportData.severity_breakdown?.warning || 0} warnings
• AI suggestions: ${reportData.ai_suggestions_included ? 'Included' : 'Not included'}`;

            vscode.window.showInformationMessage(summary);
        });

    } catch (error: any) {
        vscode.window.showErrorMessage(`Report generation failed: ${error.message}`);
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
                // Apply the fix using targeted replacement
                await applyTargetedFix(activeEditor, improvement, diagnostic);
                
                // Clear the specific diagnostic
                diagnosticsManager.clearSpecificDiagnostic(activeEditor.document.uri, diagnostic);
                
                vscode.window.showInformationMessage('ChatGPT fix applied successfully');
            }
        });

    } catch (error) {
        vscode.window.showErrorMessage(`Failed to apply ChatGPT fix: ${error}`);
    }
}

async function showBulkFixMenu() {
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

    // Group diagnostics by type
    const issueGroups: { [key: string]: vscode.Diagnostic[] } = {};
    diagnostics.forEach(diagnostic => {
        const type = diagnostic.source || 'unknown';
        if (!issueGroups[type]) {
            issueGroups[type] = [];
        }
        issueGroups[type].push(diagnostic);
    });

    // Create quick pick items for each issue type
    const groupItems = Object.keys(issueGroups).map(type => ({
        label: `${type} (${issueGroups[type].length} instances)`,
        description: `Fix all ${issueGroups[type].length} instances of ${type} issues`,
        detail: `Lines: ${issueGroups[type].map(d => d.range.start.line + 1).join(', ')}`,
        type: type,
        count: issueGroups[type].length,
        diagnostics: issueGroups[type]
    }));

    const selected = await vscode.window.showQuickPick(groupItems, {
        placeHolder: 'Select issue type to fix all instances with AI',
        matchOnDescription: true,
        matchOnDetail: true
    });

    if (selected) {
        await applyBulkFixForType(selected.type, selected.diagnostics);
    }
}

async function applyBulkFixForType(fixType: string, diagnostics: vscode.Diagnostic[]) {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) return;

    try {
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: `Bulk Fix: ${fixType}`,
            cancellable: false
        }, async (progress) => {
            progress.report({ increment: 0, message: `Analyzing ${diagnostics.length} instances...` });

            const filename = activeEditor.document.fileName.split('/').pop() || 'untitled.py';
            const content = activeEditor.document.getText();

            // Convert diagnostics to issues for the API
            const issues = diagnostics.map(diagnostic => ({
                type: diagnostic.source || fixType,
                severity: diagnostic.severity === vscode.DiagnosticSeverity.Error ? 'error' : 'warning',
                line: diagnostic.range.start.line + 1,
                column: diagnostic.range.start.character + 1,
                description: diagnostic.message,
                source: diagnostic.source,
                code: diagnostic.code
            }));

            progress.report({ increment: 30, message: `Getting AI fixes for ${fixType}...` });

            // Get bulk fixes from API
            const improvement = await api.bulkFix(content, filename, fixType, issues);
            
            progress.report({ increment: 80, message: "Applying bulk fixes..." });

            // Show confirmation dialog
            const apply = await vscode.window.showInformationMessage(
                `Bulk Fix Results:\n${improvement.improvement_summary}\n\nFixed ${improvement.instances_fixed} instances on lines: ${improvement.fixed_lines?.join(', ')}\n\nConfidence: ${Math.round(improvement.confidence_score * 100)}%`,
                'Preview Changes', 'Apply All Fixes', 'Dismiss'
            );

            if (apply === 'Preview Changes') {
                // Show diff view
                const doc = await vscode.workspace.openTextDocument({
                    content: improvement.improved_code,
                    language: 'python'
                });
                await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
                
            } else if (apply === 'Apply All Fixes') {
                // Apply all the fixes
                const edit = new vscode.WorkspaceEdit();
                const fullRange = new vscode.Range(0, 0, activeEditor.document.lineCount, 0);
                edit.replace(activeEditor.document.uri, fullRange, improvement.improved_code);
                await vscode.workspace.applyEdit(edit);
                
                // Clear all diagnostics of this type
                diagnostics.forEach(diagnostic => {
                    diagnosticsManager.clearSpecificDiagnostic(activeEditor.document.uri, diagnostic);
                });
                
                vscode.window.showInformationMessage(`Successfully applied ${improvement.instances_fixed} ${fixType} fixes`);
            }

            progress.report({ increment: 100, message: "Complete!" });
        });

    } catch (error) {
        vscode.window.showErrorMessage(`Failed to apply bulk fixes: ${error}`);
    }
}

async function applyTargetedFix(editor: vscode.TextEditor, improvement: any, diagnostic: vscode.Diagnostic) {
    try {
        const originalLines = editor.document.getText().split('\n');
        const improvedLines = improvement.improved_code.split('\n');
        
        // Find the difference and apply only the changed lines
        const lineNumber = diagnostic.range.start.line;
        
        // Check if we can apply a line-specific fix
        if (improvedLines.length === originalLines.length) {
            // Line count matches - replace only changed lines around the issue
            const startLine = Math.max(0, lineNumber - 2);
            const endLine = Math.min(originalLines.length - 1, lineNumber + 2);
            
            let hasChanges = false;
            for (let i = startLine; i <= endLine; i++) {
                if (originalLines[i] !== improvedLines[i]) {
                    hasChanges = true;
                    break;
                }
            }
            
            if (hasChanges) {
                const edit = new vscode.WorkspaceEdit();
                const range = new vscode.Range(startLine, 0, endLine + 1, 0);
                const replacementText = improvedLines.slice(startLine, endLine + 1).join('\n') + '\n';
                edit.replace(editor.document.uri, range, replacementText);
                await vscode.workspace.applyEdit(edit);
                return;
            }
        }
        
        // Fallback: show diff and let user decide
        const action = await vscode.window.showWarningMessage(
            'This fix requires significant changes. Review before applying.',
            'Show Diff', 'Replace All', 'Cancel'
        );
        
        if (action === 'Show Diff') {
            const doc = await vscode.workspace.openTextDocument({
                content: improvement.improved_code,
                language: 'python'
            });
            await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
        } else if (action === 'Replace All') {
            const edit = new vscode.WorkspaceEdit();
            const fullRange = new vscode.Range(0, 0, editor.document.lineCount, 0);
            edit.replace(editor.document.uri, fullRange, improvement.improved_code);
            await vscode.workspace.applyEdit(edit);
        }
        
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to apply targeted fix: ${error}`);
    }
}

export function deactivate() {
    if (diagnosticsManager) {
        diagnosticsManager.dispose();
    }
}