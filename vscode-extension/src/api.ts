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

    // Alias method for backwards compatibility with extension
    async auditCode(files: CodeFile[], options?: AuditOptions): Promise<AuditResponse> {
        return this.audit(files, options);
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
}