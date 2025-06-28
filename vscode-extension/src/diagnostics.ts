import * as vscode from 'vscode';
import { Issue, Fix } from './api';

export class DiagnosticsManager {
    public diagnosticCollection: vscode.DiagnosticCollection;
    private fixes: Map<string, Fix[]> = new Map();
    
    constructor() {
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('codeguard');
    }
    
    addDiagnostics(uri: vscode.Uri, issues: Issue[], fixes: Fix[]) {
        const diagnostics: vscode.Diagnostic[] = [];
        
        // Store fixes for this file
        this.fixes.set(uri.toString(), fixes);
        
        for (const issue of issues) {
            const line = Math.max(0, issue.line - 1); // Convert to 0-based
            const column = Math.max(0, issue.column);
            
            const range = new vscode.Range(
                new vscode.Position(line, column),
                new vscode.Position(line, column + 10) // Highlight a few characters
            );
            
            const diagnostic = new vscode.Diagnostic(
                range,
                this.formatDiagnosticMessage(issue),
                this.mapSeverity(issue.severity)
            );
            
            // Add additional properties
            diagnostic.source = 'CodeGuard';
            diagnostic.code = issue.rule_id || issue.type;
            
            diagnostics.push(diagnostic);
        }
        
        this.diagnosticCollection.set(uri, diagnostics);
        
        // Register code actions for fixes
        this.registerCodeActions(uri, fixes);
    }
    
    private formatDiagnosticMessage(issue: Issue): string {
        const sourceTag = issue.source ? `[${issue.source}]` : '';
        const ruleTag = issue.rule_id ? `[${issue.rule_id}]` : '';
        
        return `${sourceTag}${ruleTag} ${issue.description}`;
    }
    
    private mapSeverity(severity: string): vscode.DiagnosticSeverity {
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
    
    private registerCodeActions(uri: vscode.Uri, fixes: Fix[]) {
        // Register code action provider for this file's fixes
        const provider = vscode.languages.registerCodeActionsProvider(
            { scheme: 'file', language: 'python' },
            new CodeGuardCodeActionProvider(fixes),
            {
                providedCodeActionKinds: [vscode.CodeActionKind.QuickFix]
            }
        );
    }
    
    clearForFile(uri: vscode.Uri) {
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
    
    getFixesForFile(uri: vscode.Uri): Fix[] {
        return this.fixes.get(uri.toString()) || [];
    }
    
    getDiagnosticsForFile(uri: vscode.Uri): vscode.Diagnostic[] {
        return [...(this.diagnosticCollection.get(uri) || [])];
    }
    
    clearSpecificDiagnostic(uri: vscode.Uri, diagnostic: vscode.Diagnostic) {
        const currentDiagnostics = this.diagnosticCollection.get(uri) || [];
        const filteredDiagnostics = currentDiagnostics.filter(d => 
            d.range.start.line !== diagnostic.range.start.line ||
            d.message !== diagnostic.message
        );
        this.diagnosticCollection.set(uri, filteredDiagnostics);
    }
}

class CodeGuardCodeActionProvider implements vscode.CodeActionProvider {
    private fixes: Fix[];
    
    constructor(fixes: Fix[]) {
        this.fixes = fixes;
    }
    
    provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range | vscode.Selection,
        context: vscode.CodeActionContext,
        token: vscode.CancellationToken
    ): vscode.ProviderResult<(vscode.Command | vscode.CodeAction)[]> {
        const actions: vscode.CodeAction[] = [];
        
        // Find fixes that apply to the current line
        const currentLine = range.start.line + 1; // Convert to 1-based
        const applicableFixes = this.fixes.filter(fix => fix.line === currentLine);
        
        for (const fix of applicableFixes) {
            if (fix.auto_fixable && fix.replacement_code) {
                const action = new vscode.CodeAction(
                    `CodeGuard: ${fix.description}`,
                    vscode.CodeActionKind.QuickFix
                );
                
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