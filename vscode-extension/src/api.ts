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
            baseURL: this.configManager.getServerUrl()
        });
    }

    async audit(files: CodeFile[], options?: AuditOptions): Promise<AuditResponse> {
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

    

    async auditWithoutFilter(files: CodeFile[], options?: AuditOptions): Promise<AuditResponse> {
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
    
    async bulkFix(originalCode: string, filename: string, fixType: string, issues: Issue[]): Promise<any> {
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

    

    // Project template methods for VS Code extension
    async getProjectTemplates(): Promise<any[]> {
        const response = await this.client.get('/templates');
        return response.data;
    }

    async previewProject(templateName: string): Promise<any> {
        const response = await this.client.get(`/templates/preview?template=${templateName}`);
        return response.data;
    }

    async generateProject(templateName: string, projectPath: string): Promise<any> {
        const response = await this.client.post('/templates/generate', {
            template_name: templateName,
            project_path: projectPath
        });
        return response.data;
    }

    // GitHub context methods
    async analyzeRepository(repoUrl: string, githubToken?: string): Promise<any> {
        const requestData: any = { repo_url: repoUrl };
        if (githubToken) {
            requestData.github_token = githubToken;
        }
        
        const response = await this.client.post('/repo/analyze', requestData, {
            timeout: 120000 // 2 minute timeout for repository analysis
        });
        return response.data;
    }

    async improveWithRepositoryContext(repoUrl: string, content: string, filename: string, relativePath: string, githubToken?: string): Promise<any> {
        const aiProvider = this.configManager.getAiProvider();
        const aiApiKey = await this.configManager.getCurrentAiApiKey();
        
        const requestData: any = {
            repo_url: repoUrl,
            file_content: content,
            filename: filename,
            target_file_path: relativePath,
            ai_provider: aiProvider,
            ai_api_key: aiApiKey
        };
        
        if (githubToken) {
            requestData.github_token = githubToken;
        }
        
        const response = await this.client.post('/improve/with-related-context', requestData, {
            timeout: 180000 // 3 minute timeout for context-aware improvements
        });
        return response.data;
    }

    // System management methods
    async getCacheStats(): Promise<any> {
        const response = await this.client.get('/cache/stats');
        return response.data;
    }

    async clearCache(): Promise<any> {
        const response = await this.client.post('/cache/clear');
        return response.data;
    }

    async getRuleConfiguration(): Promise<any> {
        const response = await this.client.get('/rules/config');
        return response.data;
    }

    async toggleRuleSet(ruleSetName: string, enabled: boolean): Promise<any> {
        const response = await this.client.post(`/rules/rule-set/${ruleSetName}/toggle`, {
            enabled
        });
        return response.data;
    }

    async getSystemHealth(): Promise<any> {
        const response = await this.client.get('/system/health/detailed');
        return response.data;
    }

    // Legacy method compatibility
    async auditCode(files: CodeFile[], options?: AuditOptions): Promise<AuditResponse> {
        return this.audit(files, options);
    }

    async generateReport(files: CodeFile[], format: string = 'markdown'): Promise<any> {
        return this.generateImprovementReport(files, format, false, true);
    }
}