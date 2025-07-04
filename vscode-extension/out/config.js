"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ConfigManager = void 0;
const vscode = require("vscode");
class ConfigManager {
    constructor(context) {
        this.context = context;
    }
    async getApiKey() {
        // First try to get from secure storage
        const secretKey = await this.context.secrets.get(ConfigManager.API_KEY_SECRET);
        if (secretKey) {
            return secretKey;
        }
        // Fallback to configuration (for backwards compatibility)
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get('apiKey');
    }
    async setApiKey(apiKey) {
        await this.context.secrets.store(ConfigManager.API_KEY_SECRET, apiKey);
        // Also update configuration to empty (migrate from old storage)
        const config = vscode.workspace.getConfiguration('codeguard');
        await config.update('apiKey', '', vscode.ConfigurationTarget.Global);
    }
    getServerUrl() {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get('serverUrl') || 'https://codeguard.replit.app';
    }
    getAuditOnSave() {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get('auditOnSave') ?? true;
    }
    getFalsePositiveFiltering() {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get('enableFalsePositiveFiltering') ?? true;
    }
    async setFalsePositiveFiltering(enabled) {
        const config = vscode.workspace.getConfiguration('codeguard');
        await config.update('enableFalsePositiveFiltering', enabled, vscode.ConfigurationTarget.Global);
    }
    getFalsePositiveTimeout() {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get('falsePositiveTimeout') ?? 45;
    }
    getIgnoreRules() {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get('ignoreRules') || [];
    }
    getAnalysisLevel() {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get('analysisLevel') || 'standard';
    }
    async isConfigured() {
        const apiKey = await this.getApiKey();
        return !!(apiKey && apiKey.trim().length > 0);
    }
    async promptForApiKey() {
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
    getAiProvider() {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get('aiProvider') || 'openai';
    }
    async getOpenAiApiKey() {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get('openaiApiKey');
    }
    async getGeminiApiKey() {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get('geminiApiKey');
    }
    async getClaudeApiKey() {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get('claudeApiKey');
    }
    async getDeepSeekApiKey() {
        const config = vscode.workspace.getConfiguration('codeguard');
        return config.get('deepseekApiKey');
    }
    async getCurrentAiApiKey() {
        const provider = this.getAiProvider();
        switch (provider) {
            case 'openai':
                return this.getOpenAiApiKey();
            case 'deepseek':
                return this.getDeepSeekApiKey();
            case 'gemini':
                return this.getGeminiApiKey();
            case 'claude':
                return this.getClaudeApiKey();
            default:
                return this.getOpenAiApiKey();
        }
    }
}
exports.ConfigManager = ConfigManager;
ConfigManager.API_KEY_SECRET = 'codeguard.apiKey';
//# sourceMappingURL=config.js.map