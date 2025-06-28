"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CodeGuardAPI = void 0;
const axios_1 = require("axios");
class CodeGuardAPI {
    constructor(configManager) {
        this.configManager = configManager;
        this.client = axios_1.default.create({
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
        // Add request interceptor to add auth header
        this.client.interceptors.request.use(async (config) => {
            const apiKey = await this.configManager.getApiKey();
            const serverUrl = this.configManager.getServerUrl();
            if (apiKey) {
                config.headers['Authorization'] = `Bearer ${apiKey}`;
            }
            config.baseURL = serverUrl;
            return config;
        });
        // Add response interceptor for error handling
        this.client.interceptors.response.use((response) => response, (error) => {
            if (error.response?.status === 401) {
                throw new Error('Invalid API key. Please check your CodeGuard configuration.');
            }
            else if (error.response?.status === 429) {
                throw new Error('Rate limit exceeded. Please try again later.');
            }
            else if (error.code === 'ECONNREFUSED') {
                throw new Error('Cannot connect to CodeGuard server. Please check your server URL.');
            }
            throw error;
        });
    }
    async auditCode(files, options) {
        const requestData = {
            files,
            options: {
                level: options?.level || this.configManager.getAnalysisLevel(),
                framework: options?.framework || 'auto',
                target: options?.target || 'gpu'
            }
        };
        const response = await this.client.post('/audit', requestData);
        return response.data;
    }
    async getRulesSummary() {
        const response = await this.client.get('/rules/summary');
        return response.data;
    }
    async getRulesByTag(tag) {
        const response = await this.client.get(`/rules/by-tag/${tag}`);
        return response.data;
    }
    async getMetrics(days = 7) {
        const response = await this.client.get(`/metrics/usage?days=${days}`);
        return response.data;
    }
    async generateReport(days = 7, format = 'markdown') {
        const response = await this.client.get(`/dashboard/export?days=${days}&format=${format}`);
        return response.data;
    }
    async queryAudits(query) {
        const response = await this.client.post('/query/audits', { query });
        return response.data;
    }
    async explainIssue(issue, context) {
        const response = await this.client.post('/explain/issue', { issue, context });
        return response.data;
    }
    async getTimeline(days = 30) {
        const response = await this.client.get(`/timeline?days=${days}`);
        return response.data;
    }
    async checkHealth() {
        try {
            const response = await this.client.get('/health');
            return response.status === 200;
        }
        catch {
            return false;
        }
    }
    async getProjectTemplates() {
        const response = await this.client.get('/templates');
        return response.data.templates;
    }
    async getTemplateDetails(templateName) {
        const response = await this.client.get(`/templates/${templateName}`);
        return response.data.template;
    }
    async previewProject(templateName) {
        const response = await this.client.post('/templates/preview', { template: templateName });
        return response.data.preview;
    }
    async generateProject(templateName, projectPath, config) {
        const response = await this.client.post('/templates/generate', {
            template: templateName,
            project_path: projectPath,
            config: config || {}
        });
        return response.data.project;
    }
    async improveCode(originalCode, filename, issues, fixes) {
        const response = await this.client.post('/improve/code', {
            original_code: originalCode,
            filename,
            issues,
            fixes,
            improvement_level: 'moderate',
            preserve_functionality: true
        });
        return response.data;
    }
    async improveProject(files, auditResults) {
        const response = await this.client.post('/improve/project', {
            files,
            audit_results: auditResults
        });
        return response.data;
    }
    async auditAndImprove(files, options) {
        const requestData = {
            files,
            options: {
                level: options?.level || this.configManager.getAnalysisLevel(),
                framework: options?.framework || 'auto',
                target: options?.target || 'gpu'
            }
        };
        const response = await this.client.post('/audit-and-improve', requestData);
        return response.data;
    }
}
exports.CodeGuardAPI = CodeGuardAPI;
//# sourceMappingURL=api.js.map