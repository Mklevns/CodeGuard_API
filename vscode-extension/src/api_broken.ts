import axios, { AxiosInstance } from 'axios';
import { ConfigManager } from './config';

export interface CodeFile {
    filename: string;
    content: string;
}

export interface AuditOptions {
    level?: string;
    framework?: string;
    target?: string;
}

export interface Issue {
    line: number;
    column: number;
    type: string;
    description: string;
    severity: string;
    source: string;
    rule_id?: string;
}

export interface Fix {
    line: number;
    description: string;
    replacement_code?: string;
    diff?: string;
    auto_fixable: boolean;
}

export interface AuditResponse {
    summary: {
        total_issues: number;
        total_files: number;
        analysis_tools: string[];
    };
    issues: Issue[];
    fixes: Fix[];
}

export class CodeGuardAPI {
    private client: AxiosInstance;
    private configManager: ConfigManager;
    
    constructor(configManager: ConfigManager) {
        this.configManager = configManager;
        this.client = axios.create({
            timeout: 60000, // Increased timeout for ChatGPT false positive filtering
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
        this.client.interceptors.response.use(
            (response) => response,
            (error) => {
                if (error.response?.status === 401) {
                    throw new Error('Invalid API key. Please check your CodeGuard configuration.');
                } else if (error.response?.status === 429) {
                    throw new Error('Rate limit exceeded. Please try again later.');
                } else if (error.code === 'ECONNREFUSED') {
                    throw new Error('Cannot connect to CodeGuard server. Please check your server URL.');
                }
                throw error;
            }
        );
    }
    
    async auditCode(files: CodeFile[], options?: AuditOptions): Promise<AuditResponse> {
        const requestData = {
            files,
            options: {
                level: options?.level || this.configManager.getAnalysisLevel(),
                framework: options?.framework || 'auto',
                target: options?.target || 'gpu'
            }
        };
        
        // Use appropriate endpoint based on false positive filtering setting
        const useFalsePositiveFiltering = this.configManager.getFalsePositiveFiltering();
        const endpoint = useFalsePositiveFiltering ? '/audit' : '/audit/no-filter';
        
        // Get user-configured timeout for false positive filtering
        const falsePositiveTimeout = this.configManager.getFalsePositiveTimeout();
        const timeout = useFalsePositiveFiltering ? falsePositiveTimeout * 1000 : 30000;
        
        try {
            const response = await this.client.post(endpoint, requestData, { timeout });
            return response.data;
        } catch (error: any) {
            // If timeout with false positive filtering enabled, try without filtering
            if (error.code === 'ECONNABORTED' && useFalsePositiveFiltering) {
                console.log(`AI filtering timed out after ${falsePositiveTimeout}s, retrying without filtering...`);
                const fallbackResponse = await this.client.post('/audit/no-filter', requestData, { timeout: 30000 });
                return {
                    ...fallbackResponse.data,
                    summary: fallbackResponse.data.summary + ' (AI filtering timed out - using standard analysis)'
                };
            }
            throw error;
        }
    }
    
    async getRulesSummary(): Promise<any> {
        const response = await this.client.get('/rules/summary');
        return response.data;
    }
    
    async getRulesByTag(tag: string): Promise<any> {
        const response = await this.client.get(`/rules/by-tag/${tag}`);
        return response.data;
    }
    
    async getMetrics(days: number = 7): Promise<any> {
        const response = await this.client.get(`/metrics/usage?days=${days}`);
        return response.data;
    }
    
    async generateReport(days: number = 7, format: string = 'markdown'): Promise<any> {
        const response = await this.client.get(`/dashboard/export?days=${days}&format=${format}`);
        return response.data;
    }
    
    async queryAudits(query: string): Promise<any> {
        const response = await this.client.post('/query/audits', { query });
        return response.data;
    }
    
    async explainIssue(issue: string, context?: string): Promise<any> {
        const response = await this.client.post('/explain/issue', { issue, context });
        return response.data;
    }
    
    async getTimeline(days: number = 30): Promise<any> {
        const response = await this.client.get(`/timeline?days=${days}`);
        return response.data;
    }
    
    async checkHealth(): Promise<boolean> {
        try {
            const response = await this.client.get('/health');
            return response.status === 200;
        } catch {
            return false;
        }
    }
    
    async getProjectTemplates(): Promise<any[]> {
        const response = await this.client.get('/templates');
        return response.data.templates;
    }
    
    async getTemplateDetails(templateName: string): Promise<any> {
        const response = await this.client.get(`/templates/${templateName}`);
        return response.data.template;
    }
    
    async previewProject(templateName: string): Promise<any> {
        const response = await this.client.post('/templates/preview', { template: templateName });
        return response.data.preview;
    }
    
    async generateProject(templateName: string, projectPath: string, config?: any): Promise<any> {
        const response = await this.client.post('/templates/generate', {
            template: templateName,
            project_path: projectPath,
            config: config || {}
        });
        return response.data.project;
    }
    
    async improveCode(originalCode: string, filename: string, issues: Issue[], fixes: Fix[]): Promise<any> {
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
    
    async improveProject(files: CodeFile[]): Promise<any> {
        const response = await this.client.post('/improve/project', {
            files,
            improvement_level: 'moderate'
        }, { timeout: 180000 }); // 3 minute timeout for project-wide improvements
        
        return response.data;
    }
    
    async auditAndImprove(files: CodeFile[], options?: AuditOptions): Promise<any> {
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
    
    async bulkFixByType(originalCode: string, filename: string, fixType: string, issues: Issue[]): Promise<any> {
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
    
    async generateImprovementReport(files: CodeFile[], format: string = 'markdown', includeAiSuggestions: boolean = true, applyFiltering: boolean = true): Promise<any> {
        const response = await this.client.post('/reports/improvement-analysis', {
            files,
            format,
            include_ai_suggestions: includeAiSuggestions,
            apply_false_positive_filtering: applyFiltering
        }, { timeout: 120000 }); // 2 minute timeout for report generation
        
        return response.data;
    }


}