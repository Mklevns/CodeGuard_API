"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CodeGuardAPI = void 0;
const axios_1 = require("axios");
class CodeGuardAPI {
    constructor(configManager) {
        this.configManager = configManager;
        this.client = axios_1.default.create({
            timeout: 60000, // Increased timeout for ChatGPT false positive filtering
            baseURL: this.configManager.getServerUrl()
        });
    }
    async audit(files, options) {
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
    // Alias method for backwards compatibility with extension
    async auditCode(files, options) {
        return this.audit(files, options);
    }
    async auditWithoutFilter(files, options) {
        const requestData = {
            files,
            options: {
                level: options?.level || this.configManager.getAnalysisLevel(),
                framework: options?.framework || 'auto',
                target: options?.target || 'gpu'
            }
        };
        const response = await this.client.post('/audit/no-filter', requestData);
        return response.data;
    }
    async improveCode(originalCode, filename, issues, fixes) {
        const aiProvider = this.configManager.getAiProvider();
        const aiApiKey = await this.configManager.getCurrentAiApiKey();
        const response = await this.client.post('/improve/code', {
            original_code: originalCode,
            filename,
            issues,
            fixes,
            improvement_level: 'moderate',
            preserve_functionality: true,
            ai_provider: aiProvider,
            ai_api_key: aiApiKey
        }, { timeout: 120000 }); // 2 minute timeout for AI improvements
        return response.data;
    }
    async improveProject(files) {
        const response = await this.client.post('/improve/project', {
            files,
            improvement_level: 'moderate'
        }, { timeout: 180000 }); // 3 minute timeout for project-wide improvements
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
        const response = await this.client.post('/audit-and-improve', requestData, {
            timeout: 150000 // 2.5 minute timeout for combined workflow
        });
        return response.data;
    }
    async bulkFix(originalCode, filename, fixType, issues) {
        const aiProvider = this.configManager.getAiProvider();
        const aiApiKey = await this.configManager.getCurrentAiApiKey();
        const response = await this.client.post('/improve/bulk-fix', {
            original_code: originalCode,
            filename,
            fix_type: fixType,
            issues,
            ai_provider: aiProvider,
            ai_api_key: aiApiKey
        }, { timeout: 90000 }); // 1.5 minute timeout for bulk fixes
        return response.data;
    }
    async generateImprovementReport(files, format = 'markdown', includeAiSuggestions = true, applyFiltering = true) {
        const response = await this.client.post('/reports/improvement-analysis', {
            files,
            format,
            include_ai_suggestions: includeAiSuggestions,
            apply_false_positive_filtering: applyFiltering
        }, { timeout: 120000 }); // 2 minute timeout for report generation
        return response.data;
    }
    // Alias method for backwards compatibility with extension
    async generateReport(files, format = 'markdown', includeAiSuggestions = true, applyFiltering = true) {
        return this.generateImprovementReport(files, format, includeAiSuggestions, applyFiltering);
    }
    // Project template methods for VS Code extension
    async getProjectTemplates() {
        const response = await this.client.get('/templates');
        return response.data;
    }
    async previewProject(templateName) {
        const response = await this.client.get(`/templates/preview?template=${templateName}`);
        return response.data;
    }
    async generateProject(templateName, projectPath) {
        const response = await this.client.post('/templates/generate', {
            template_name: templateName,
            project_path: projectPath
        });
        return response.data;
    }
}
exports.CodeGuardAPI = CodeGuardAPI;
//# sourceMappingURL=api.js.map