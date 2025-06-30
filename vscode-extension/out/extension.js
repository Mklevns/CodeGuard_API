"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = require("vscode");
const api_1 = require("./api");
const diagnostics_1 = require("./diagnostics");
const config_1 = require("./config");
const project_setup_1 = require("./project_setup");
let diagnosticsManager;
let api;
let configManager;
let projectSetupManager;
function activate(context) {
    // Initialize components
    configManager = new config_1.ConfigManager(context);
    api = new api_1.CodeGuardAPI(configManager);
    diagnosticsManager = new diagnostics_1.DiagnosticsManager();
    projectSetupManager = new project_setup_1.ProjectSetupManager(api);
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
    const analyzeRepositoryCommand = vscode.commands.registerCommand('codeguard.analyzeRepository', async () => {
        await analyzeCurrentRepository();
    });
    const improveWithContextCommand = vscode.commands.registerCommand('codeguard.improveWithContext', async () => {
        await improveWithRepositoryContext();
    });
    const showCacheStatsCommand = vscode.commands.registerCommand('codeguard.showCacheStats', async () => {
        await showCacheStatistics();
    });
    const clearCacheCommand = vscode.commands.registerCommand('codeguard.clearCache', async () => {
        await clearAnalysisCache();
    });
    const configureRulesCommand = vscode.commands.registerCommand('codeguard.configureRules', async () => {
        await showRuleConfiguration();
    });
    const systemHealthCommand = vscode.commands.registerCommand('codeguard.systemHealth', async () => {
        await showSystemHealth();
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
    context.subscriptions.push(runAuditCommand, clearDiagnosticsCommand, generateReportCommand, setupProjectCommand, applySingleFixCommand, improveWithChatGPTCommand, showFixMenuCommand, bulkFixCommand, onSaveListener, diagnosticsManager.diagnosticCollection);
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
async function runAuditForDocument(document) {
    if (!await configManager.isConfigured()) {
        const configure = await vscode.window.showErrorMessage('CodeGuard API key not configured. Would you like to configure it now?', 'Configure');
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
            vscode.window.showInformationMessage(`CodeGuard: Found ${issueCount} issues, ${fixCount} auto-fixable`);
        });
    }
    catch (error) {
        vscode.window.showErrorMessage(`CodeGuard analysis failed: ${error}`);
    }
}
async function generateReport() {
    try {
        const activeEditor = vscode.window.activeTextEditor;
        if (!activeEditor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }
        const document = activeEditor.document;
        if (!document.fileName.endsWith('.py')) {
            vscode.window.showErrorMessage('Please open a Python file to generate a report');
            return;
        }
        const result = await vscode.window.showQuickPick([
            { label: 'Markdown Report', value: 'markdown' },
            { label: 'JSON Report', value: 'json' }
        ], {
            placeHolder: 'Select report format'
        });
        if (!result)
            return;
        const files = [{
                filename: document.fileName.split('/').pop() || 'untitled.py',
                content: document.getText()
            }];
        const reportData = await api.generateReport(files, result.value);
        // Create new document with report
        const doc = await vscode.workspace.openTextDocument({
            content: reportData.report,
            language: result.value === 'markdown' ? 'markdown' : 'json'
        });
        await vscode.window.showTextDocument(doc);
    }
    catch (error) {
        vscode.window.showErrorMessage(`Failed to generate report: ${error}`);
    }
}
async function applySingleFix(fix) {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showWarningMessage('No active file to apply fix');
        return;
    }
    try {
        if (fix.replacement_code) {
            // Apply the fix directly
            const edit = new vscode.WorkspaceEdit();
            const range = new vscode.Range(fix.line - 1, 0, fix.line - 1, activeEditor.document.lineAt(fix.line - 1).text.length);
            edit.replace(activeEditor.document.uri, range, fix.replacement_code);
            await vscode.workspace.applyEdit(edit);
            vscode.window.showInformationMessage(`Applied fix: ${fix.description}`);
        }
        else {
            // Show fix details for manual application
            const apply = await vscode.window.showInformationMessage(`Fix suggestion: ${fix.description}\n\nLine ${fix.line}: ${fix.suggestion}`, 'Apply Manual Fix', 'Dismiss');
            if (apply === 'Apply Manual Fix') {
                // Navigate to the line
                const position = new vscode.Position(fix.line - 1, 0);
                activeEditor.selection = new vscode.Selection(position, position);
                activeEditor.revealRange(new vscode.Range(position, position));
            }
        }
    }
    catch (error) {
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
            const applyChanges = await vscode.window.showInformationMessage(`ChatGPT Improvements Applied:\n${improvement.improvement_summary}\n\nConfidence: ${Math.round(improvement.confidence_score * 100)}%`, 'Replace Original', 'Keep Both', 'Dismiss');
            if (applyChanges === 'Replace Original') {
                // Replace the original file content
                const edit = new vscode.WorkspaceEdit();
                const fullRange = new vscode.Range(0, 0, activeEditor.document.lineCount, 0);
                edit.replace(activeEditor.document.uri, fullRange, improvement.improved_code);
                await vscode.workspace.applyEdit(edit);
                vscode.window.showInformationMessage('Original file updated with ChatGPT improvements');
            }
        });
    }
    catch (error) {
        vscode.window.showErrorMessage(`ChatGPT improvement failed: ${error}`);
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
        const format = await vscode.window.showQuickPick(['Markdown', 'HTML', 'JSON'], { placeHolder: 'Select report format' });
        if (!format)
            return;
        // Show AI inclusion option
        const includeAi = await vscode.window.showQuickPick(['Yes', 'No'], { placeHolder: 'Include AI improvement suggestions?' });
        if (!includeAi)
            return;
        // Show filtering option
        const applyFiltering = await vscode.window.showQuickPick(['Yes - Filter out style issues', 'No - Show all issues'], { placeHolder: 'Apply false positive filtering?' });
        if (!applyFiltering)
            return;
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
            const reportData = await api.generateImprovementReport(files, format.toLowerCase(), includeAi === 'Yes', applyFiltering.startsWith('Yes'));
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
    }
    catch (error) {
        vscode.window.showErrorMessage(`Report generation failed: ${error.message}`);
    }
}
async function applySpecificChatGPTFix(diagnostic) {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor)
        return;
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
            const apply = await vscode.window.showInformationMessage(`ChatGPT Fix for Line ${line}:\n${improvement.improvement_summary}`, 'Preview Changes', 'Apply Fix', 'Dismiss');
            if (apply === 'Preview Changes') {
                // Show diff view
                const doc = await vscode.workspace.openTextDocument({
                    content: improvement.improved_code,
                    language: 'python'
                });
                await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
            }
            else if (apply === 'Apply Fix') {
                // Apply the fix using targeted replacement
                await applyTargetedFix(activeEditor, improvement, diagnostic);
                // Clear the specific diagnostic
                diagnosticsManager.clearSpecificDiagnostic(activeEditor.document.uri, diagnostic);
                vscode.window.showInformationMessage('ChatGPT fix applied successfully');
            }
        });
    }
    catch (error) {
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
    const issueGroups = {};
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
async function applyBulkFixForType(fixType, diagnostics) {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor)
        return;
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
            const apply = await vscode.window.showInformationMessage(`Bulk Fix Results:\n${improvement.improvement_summary}\n\nFixed ${improvement.instances_fixed} instances on lines: ${improvement.fixed_lines?.join(', ')}\n\nConfidence: ${Math.round(improvement.confidence_score * 100)}%`, 'Preview Changes', 'Apply All Fixes', 'Dismiss');
            if (apply === 'Preview Changes') {
                // Show diff view
                const doc = await vscode.workspace.openTextDocument({
                    content: improvement.improved_code,
                    language: 'python'
                });
                await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
            }
            else if (apply === 'Apply All Fixes') {
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
    }
    catch (error) {
        vscode.window.showErrorMessage(`Failed to apply bulk fixes: ${error}`);
    }
}
async function applyTargetedFix(editor, improvement, diagnostic) {
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
        const action = await vscode.window.showWarningMessage('This fix requires significant changes. Review before applying.', 'Show Diff', 'Replace All', 'Cancel');
        if (action === 'Show Diff') {
            const doc = await vscode.workspace.openTextDocument({
                content: improvement.improved_code,
                language: 'python'
            });
            await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
        }
        else if (action === 'Replace All') {
            const edit = new vscode.WorkspaceEdit();
            const fullRange = new vscode.Range(0, 0, editor.document.lineCount, 0);
            edit.replace(editor.document.uri, fullRange, improvement.improved_code);
            await vscode.workspace.applyEdit(edit);
        }
    }
    catch (error) {
        vscode.window.showErrorMessage(`Failed to apply targeted fix: ${error}`);
    }
}
async function analyzeCurrentRepository() {
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (!workspaceFolder) {
        vscode.window.showWarningMessage('No workspace folder open. Please open a Git repository.');
        return;
    }
    try {
        // Try to get Git repository URL
        const gitExtension = vscode.extensions.getExtension('vscode.git')?.exports;
        const git = gitExtension?.getAPI(1);
        const repo = git?.repositories[0];
        if (!repo) {
            vscode.window.showWarningMessage('No Git repository detected in current workspace.');
            return;
        }
        const remoteUrl = repo.state.remotes[0]?.fetchUrl;
        if (!remoteUrl || !remoteUrl.includes('github.com')) {
            vscode.window.showWarningMessage('Current repository is not a GitHub repository.');
            return;
        }
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Analyzing GitHub Repository",
            cancellable: false
        }, async (progress) => {
            progress.report({ increment: 20, message: "Connecting to repository..." });
            const analysis = await api.analyzeRepository(remoteUrl);
            progress.report({ increment: 100, message: "Analysis complete!" });
            const summary = `Repository Analysis Complete:
• Language: ${analysis.repository_info.language}
• Frameworks: ${analysis.detected_frameworks.join(', ')}
• Python Files: ${analysis.file_structure.python_files}
• Dependencies: ${analysis.file_structure.dependencies?.length || 0}
• Configuration Files: ${analysis.file_structure.config_files?.length || 0}`;
            vscode.window.showInformationMessage(summary);
        });
    }
    catch (error) {
        vscode.window.showErrorMessage(`Repository analysis failed: ${error}`);
    }
}
async function improveWithRepositoryContext() {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showWarningMessage('No active file to improve');
        return;
    }
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (!workspaceFolder) {
        vscode.window.showWarningMessage('No workspace folder open');
        return;
    }
    try {
        const gitExtension = vscode.extensions.getExtension('vscode.git')?.exports;
        const git = gitExtension?.getAPI(1);
        const repo = git?.repositories[0];
        if (!repo) {
            vscode.window.showWarningMessage('No Git repository detected. Repository context requires a Git repository.');
            return;
        }
        const remoteUrl = repo.state.remotes[0]?.fetchUrl;
        if (!remoteUrl || !remoteUrl.includes('github.com')) {
            vscode.window.showWarningMessage('Repository context requires a GitHub repository.');
            return;
        }
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Improving with Repository Context",
            cancellable: false
        }, async (progress) => {
            progress.report({ increment: 20, message: "Analyzing repository context..." });
            const filename = activeEditor.document.fileName.split('/').pop() || 'untitled.py';
            const content = activeEditor.document.getText();
            const relativePath = vscode.workspace.asRelativePath(activeEditor.document.fileName);
            progress.report({ increment: 50, message: "Getting context-aware improvements..." });
            const improvement = await api.improveWithRepositoryContext(remoteUrl, content, filename, relativePath);
            progress.report({ increment: 100, message: "Context-aware improvement complete!" });
            const apply = await vscode.window.showInformationMessage(`Context-Aware Improvement Complete:\n${improvement.improvement_summary}\n\nRelated Files Used: ${improvement.related_files_used}\nConfidence: ${Math.round(improvement.confidence_score * 100)}%`, 'Preview Changes', 'Apply Improvement', 'Dismiss');
            if (apply === 'Preview Changes') {
                const doc = await vscode.workspace.openTextDocument({
                    content: improvement.improved_code,
                    language: 'python'
                });
                await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
            }
            else if (apply === 'Apply Improvement') {
                const edit = new vscode.WorkspaceEdit();
                const fullRange = new vscode.Range(0, 0, activeEditor.document.lineCount, 0);
                edit.replace(activeEditor.document.uri, fullRange, improvement.improved_code);
                await vscode.workspace.applyEdit(edit);
                vscode.window.showInformationMessage('Context-aware improvement applied successfully');
            }
        });
    }
    catch (error) {
        vscode.window.showErrorMessage(`Context-aware improvement failed: ${error}`);
    }
}
async function showCacheStatistics() {
    try {
        const stats = await api.getCacheStats();
        const message = `Analysis Cache Statistics:
• Cache Entries: ${stats.file_cache.entries}
• Valid Entries: ${stats.file_cache.valid_entries}
• Storage Used: ${stats.file_cache.size_mb} MB
• TTL Hours: ${stats.file_cache.ttl_hours}
• Performance Impact: ${stats.performance_impact.estimated_speedup}`;
        const action = await vscode.window.showInformationMessage(message, 'Clear Cache', 'Dismiss');
        if (action === 'Clear Cache') {
            await clearAnalysisCache();
        }
    }
    catch (error) {
        vscode.window.showErrorMessage(`Failed to load cache statistics: ${error}`);
    }
}
async function clearAnalysisCache() {
    try {
        const confirm = await vscode.window.showWarningMessage('Are you sure you want to clear the analysis cache? This will remove all cached results.', 'Clear Cache', 'Cancel');
        if (confirm === 'Clear Cache') {
            await api.clearCache();
            vscode.window.showInformationMessage('Analysis cache cleared successfully');
        }
    }
    catch (error) {
        vscode.window.showErrorMessage(`Failed to clear cache: ${error}`);
    }
}
async function showRuleConfiguration() {
    try {
        const config = await api.getRuleConfiguration();
        const ruleSetOptions = Object.keys(config.rule_sets).map(name => ({
            label: `${name} (${config.rule_sets[name].length} rules)`,
            description: `Toggle ${name} rule set`,
            ruleSet: name
        }));
        const action = await vscode.window.showQuickPick([
            { label: 'View Configuration Summary', description: 'Show current rule configuration' },
            { label: 'Enable Rule Set', description: 'Enable a complete rule set' },
            { label: 'Disable Rule Set', description: 'Disable a complete rule set' },
            ...ruleSetOptions.map(opt => ({ ...opt, label: `Configure: ${opt.label}` }))
        ], {
            placeHolder: 'Select rule configuration action'
        });
        if (!action)
            return;
        if (action.label === 'View Configuration Summary') {
            const summary = `Rule Configuration:
• Total Rules: ${config.configuration.total_rules}
• Enabled Rules: ${config.configuration.enabled_rules}
• Disabled Rules: ${config.configuration.disabled_rules}

Rules by Severity:
${Object.entries(config.configuration.by_severity).map(([severity, count]) => `• ${severity}: ${count}`).join('\n')}

Available Rule Sets:
${Object.entries(config.rule_sets).map(([name, rules]) => `• ${name}: ${rules.length} rules`).join('\n')}`;
            vscode.window.showInformationMessage(summary);
        }
        else if (action.label === 'Enable Rule Set' || action.label === 'Disable Rule Set') {
            const ruleSet = await vscode.window.showQuickPick(Object.keys(config.rule_sets).map(name => ({ label: name, description: `${config.rule_sets[name].length} rules` })), { placeHolder: `Select rule set to ${action.label.includes('Enable') ? 'enable' : 'disable'}` });
            if (ruleSet) {
                const enabled = action.label.includes('Enable');
                await api.toggleRuleSet(ruleSet.label, enabled);
                vscode.window.showInformationMessage(`Rule set '${ruleSet.label}' ${enabled ? 'enabled' : 'disabled'}`);
            }
        }
    }
    catch (error) {
        vscode.window.showErrorMessage(`Failed to load rule configuration: ${error}`);
    }
}
async function showSystemHealth() {
    try {
        const health = await api.getSystemHealth();
        const status = health.status.toUpperCase();
        const message = `System Health: ${status}

Analysis Engine: ${health.analysis_engine.status}
• Tools Available: ${health.analysis_engine.tools_available}
• Semantic Analysis: ${health.analysis_engine.semantic_analysis}
• Caching: ${health.analysis_engine.caching}

Cache System: ${health.cache_system.status}
• Entries: ${health.cache_system.entries}
• Size: ${health.cache_system.size_mb} MB

Rule System: ${health.rule_system.status}
• Total Rules: ${health.rule_system.total_rules}
• Enabled Rules: ${health.rule_system.enabled_rules}

Authentication: ${health.authentication.status}
• Mode: ${health.authentication.mode}

Version: ${health.version}
Environment: ${health.environment}`;
        vscode.window.showInformationMessage(message);
    }
    catch (error) {
        vscode.window.showErrorMessage(`Failed to load system health: ${error}`);
    }
}
function deactivate() {
    if (diagnosticsManager) {
        diagnosticsManager.dispose();
    }
}
//# sourceMappingURL=extension.js.map