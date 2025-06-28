"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DiagnosticsManager = void 0;
const vscode = require("vscode");
class DiagnosticsManager {
    constructor() {
        this.fixes = new Map();
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('codeguard');
    }
    addDiagnostics(uri, issues, fixes) {
        const diagnostics = [];
        // Store fixes for this file
        this.fixes.set(uri.toString(), fixes);
        for (const issue of issues) {
            const line = Math.max(0, issue.line - 1); // Convert to 0-based
            const column = Math.max(0, issue.column);
            const range = new vscode.Range(new vscode.Position(line, column), new vscode.Position(line, column + 10) // Highlight a few characters
            );
            const diagnostic = new vscode.Diagnostic(range, this.formatDiagnosticMessage(issue), this.mapSeverity(issue.severity));
            // Add additional properties
            diagnostic.source = 'CodeGuard';
            diagnostic.code = issue.rule_id || issue.type;
            diagnostics.push(diagnostic);
        }
        this.diagnosticCollection.set(uri, diagnostics);
        // Register code actions for fixes
        this.registerCodeActions(uri, fixes);
    }
    formatDiagnosticMessage(issue) {
        const sourceTag = issue.source ? `[${issue.source}]` : '';
        const ruleTag = issue.rule_id ? `[${issue.rule_id}]` : '';
        return `${sourceTag}${ruleTag} ${issue.description}`;
    }
    mapSeverity(severity) {
        switch (severity.toLowerCase()) {
            case 'error':
                return vscode.DiagnosticSeverity.Error;
            case 'warning':
                return vscode.DiagnosticSeverity.Warning;
            case 'info':
                return vscode.DiagnosticSeverity.Information;
            default:
                return vscode.DiagnosticSeverity.Hint;
        }
    }
    registerCodeActions(uri, fixes) {
        // Register code action provider for this file's fixes
        const provider = vscode.languages.registerCodeActionsProvider({ scheme: 'file', language: 'python' }, new CodeGuardCodeActionProvider(fixes), {
            providedCodeActionKinds: [vscode.CodeActionKind.QuickFix]
        });
    }
    clearForFile(uri) {
        this.diagnosticCollection.delete(uri);
        this.fixes.delete(uri.toString());
    }
    clearAll() {
        this.diagnosticCollection.clear();
        this.fixes.clear();
    }
    dispose() {
        this.diagnosticCollection.dispose();
        this.fixes.clear();
    }
    getFixesForFile(uri) {
        return this.fixes.get(uri.toString()) || [];
    }
}
exports.DiagnosticsManager = DiagnosticsManager;
class CodeGuardCodeActionProvider {
    constructor(fixes) {
        this.fixes = fixes;
    }
    provideCodeActions(document, range, context, token) {
        const actions = [];
        // Find fixes that apply to the current line
        const currentLine = range.start.line + 1; // Convert to 1-based
        const applicableFixes = this.fixes.filter(fix => fix.line === currentLine);
        for (const fix of applicableFixes) {
            if (fix.auto_fixable && fix.replacement_code) {
                const action = new vscode.CodeAction(`CodeGuard: ${fix.description}`, vscode.CodeActionKind.QuickFix);
                action.edit = new vscode.WorkspaceEdit();
                const lineRange = document.lineAt(currentLine - 1).range;
                action.edit.replace(document.uri, lineRange, fix.replacement_code);
                action.isPreferred = true;
                actions.push(action);
            }
        }
        return actions;
    }
}
//# sourceMappingURL=diagnostics.js.map