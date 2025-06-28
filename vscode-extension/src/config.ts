import * as vscode from 'vscode';

interface CodeGuardConfig {
    serverUrl: string;
    apiKey: string;
    analysisLevel: string;
    autoAnalysisOnSave: boolean;
    enableFalsePositiveFiltering: boolean;
    aiProvider: string;
}

export class ConfigManager {
    private context: vscode.ExtensionContext;
    private static readonly API_KEY_SECRET = 'codeguard.apiKey';
    
    constructor(context: vscode.ExtensionContext) {
        this.context = context;
    }
    
    async getApiKey(): Promise<string | undefined> {
        // First try to get from secure storage
        const secretKey = await this.context.secrets.get(ConfigManager.API_KEY_SECRET);
        if (secretKey) {
            return secretKey;
        }
        
        // Fallback to configuration (for backwards compatibility)
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get<string>('apiKey');
    }
    
    async setApiKey(apiKey: string): Promise<void> {
        await this.context.secrets.store(ConfigManager.API_KEY_SECRET, apiKey);
        
        // Also update configuration to empty (migrate from old storage)
        const config = vscode.workspace.getConfiguration('codeguard');
        await config.update('apiKey', '', vscode.ConfigurationTarget.Global);
    }
    
    getServerUrl(): string {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get<string>('serverUrl') || 'https://codeguard.replit.app';
    }
    
    getAuditOnSave(): boolean {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get<boolean>('auditOnSave') ?? true;
    }
    
    getFalsePositiveFiltering(): boolean {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get<boolean>('enableFalsePositiveFiltering') ?? true;
    }
    
    async setFalsePositiveFiltering(enabled: boolean): Promise<void> {
        const config = vscode.workspace.getConfiguration('codeguard');
        await config.update('enableFalsePositiveFiltering', enabled, vscode.ConfigurationTarget.Global);
    }
    
    getIgnoreRules(): string[] {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get<string[]>('ignoreRules') || [];
    }
    
    getAnalysisLevel(): string {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get<string>('analysisLevel') || 'standard';
    }
    
    async isConfigured(): Promise<boolean> {
        const apiKey = await this.getApiKey();
        return !!(apiKey && apiKey.trim().length > 0);
    }
    
    async promptForApiKey(): Promise<void> {
        const apiKey = await vscode.window.showInputBox({
            prompt: 'Enter your CodeGuard API key',
            password: true,
            placeHolder: 'API key from CodeGuard dashboard',
            ignoreFocusOut: true,
            validateInput: (value) => {
                if (!value || value.trim().length === 0) {
                    return 'API key cannot be empty';
                }
                if (value.length < 10) {
                    return 'API key seems too short';
                }
                return null;
            }
        });
        
        if (apiKey) {
            await this.setApiKey(apiKey);
            vscode.window.showInformationMessage('CodeGuard API key saved securely');
        }
    }
    
    getAiProvider(): string {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get<string>('aiProvider') || 'openai';
    }
    
    async getOpenAiApiKey(): Promise<string | undefined> {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get<string>('openaiApiKey');
    }
    
    async getGeminiApiKey(): Promise<string | undefined> {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get<string>('geminiApiKey');
    }
    
    async getClaudeApiKey(): Promise<string | undefined> {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get<string>('claudeApiKey');
    }
    
    async getCurrentAiApiKey(): Promise<string | undefined> {
        const provider = this.getAiProvider();
        switch (provider) {
            case 'openai':
                return this.getOpenAiApiKey();
            case 'gemini':
                return this.getGeminiApiKey();
            case 'claude':
                return this.getClaudeApiKey();
            default:
                return this.getOpenAiApiKey();
        }
    }
}