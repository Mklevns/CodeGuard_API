class CodeGuardPlayground {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.currentResults = null;
        this.reportCache = new Map();
        this.debouncedGenerateReport = this.debounce(this.generateReport.bind(this), 300);
        this.init();
    }

    init() {
        this.loadSavedApiKey();
        this.setupEventListeners();
        this.loadExampleCode();
    }

    loadSavedApiKey() {
        const savedKey = localStorage.getItem('codeguard_api_key');
        const savedProvider = localStorage.getItem('codeguard_ai_provider');
        const rememberKey = localStorage.getItem('codeguard_remember_key') === 'true';
        
        if (savedKey && rememberKey) {
            document.getElementById('apiKey').value = savedKey;
            document.getElementById('rememberKey').checked = true;
        }
        
        if (savedProvider) {
            document.getElementById('aiProvider').value = savedProvider;
        }
    }

    setupEventListeners() {
        // Helper function to safely add event listeners
        const safeAddEventListener = (elementId, event, handler) => {
            const element = document.getElementById(elementId);
            if (element) {
                element.addEventListener(event, handler);
            } else {
                console.warn(`Element with ID '${elementId}' not found`);
            }
        };

        // API key management
        safeAddEventListener('rememberKey', 'change', (e) => {
            if (e.target.checked) {
                const apiKey = document.getElementById('apiKey')?.value;
                const provider = document.getElementById('aiProvider')?.value;
                if (apiKey) {
                    localStorage.setItem('codeguard_api_key', apiKey);
                    localStorage.setItem('codeguard_ai_provider', provider);
                    localStorage.setItem('codeguard_remember_key', 'true');
                }
            } else {
                localStorage.removeItem('codeguard_api_key');
                localStorage.removeItem('codeguard_ai_provider');
                localStorage.removeItem('codeguard_remember_key');
            }
        });

        safeAddEventListener('apiKey', 'input', (e) => {
            const rememberKeyEl = document.getElementById('rememberKey');
            if (rememberKeyEl?.checked) {
                localStorage.setItem('codeguard_api_key', e.target.value);
            }
        });

        safeAddEventListener('aiProvider', 'change', (e) => {
            const rememberKeyEl = document.getElementById('rememberKey');
            if (rememberKeyEl?.checked) {
                localStorage.setItem('codeguard_ai_provider', e.target.value);
            }
        });

        // Code management
        safeAddEventListener('loadExample', 'click', () => this.loadExampleCode());
        safeAddEventListener('clearCode', 'click', () => this.clearCode());

        // Repository context
        safeAddEventListener('githubRepoUrl', 'input', () => this.validateRepoUrl());
        safeAddEventListener('analyzeRepo', 'click', () => this.analyzeRepository());
        safeAddEventListener('repoFileSelect', 'change', () => this.loadSelectedRepoFile());

        // Analysis buttons
        safeAddEventListener('auditBtn', 'click', () => this.auditCode());
        safeAddEventListener('improveBtn', 'click', () => this.improveCode());
        safeAddEventListener('auditImproveBtn', 'click', () => this.auditAndImprove());
        safeAddEventListener('improveWithContextBtn', 'click', () => this.improveWithRelatedContext());
        safeAddEventListener('fimBtn', 'click', () => this.openFimTab());
        
        // FIM completion buttons
        safeAddEventListener('runFimBtn', 'click', () => this.runFimCompletion());
        safeAddEventListener('loadFimExample', 'click', () => this.loadFimExample());
        safeAddEventListener('copyFimResult', 'click', () => this.copyFimResult());
        safeAddEventListener('applyFimResult', 'click', () => this.applyFimResult());

        // Tab switching - safely handle tab buttons
        const tabButtons = document.querySelectorAll('.tab-btn');
        if (tabButtons.length > 0) {
            tabButtons.forEach(btn => {
                btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
            });
        } else {
            console.warn('No tab buttons found on page');
        }

        // Export buttons
        safeAddEventListener('copyImproved', 'click', () => this.copyImprovedCode());
        safeAddEventListener('downloadImproved', 'click', () => this.downloadImprovedCode());
        safeAddEventListener('exportMarkdown', 'click', () => this.exportReport('markdown'));
        safeAddEventListener('exportHtml', 'click', () => this.exportReport('html'));
        
        // System management buttons
        safeAddEventListener('cacheStatsBtn', 'click', () => this.showCacheStats());
        safeAddEventListener('clearCacheBtn', 'click', () => this.clearCache());
        safeAddEventListener('ruleConfigBtn', 'click', () => this.showRuleConfig());
        safeAddEventListener('systemHealthBtn', 'click', () => this.showSystemHealth());
    }

    loadExampleCode() {
        const exampleCode = `import torch
import numpy as np
import pickle

# Example ML code with various issues
def train_model(data):
    # Missing random seed
    model = torch.nn.Linear(10, 1)
    
    # Potential security issue
    config = pickle.load(open('config.pkl', 'rb'))
    
    # Missing error handling
    for epoch in range(100):
        loss = model(data)
        print(f"Loss: {loss}")  # Should use logging
        
    return model

# Unused import and undefined variable
def evaluate_model():
    results = model.evaluate(test_data)
    return results`;

        document.getElementById('codeInput').value = exampleCode;
    }

    clearCode() {
        document.getElementById('codeInput').value = '';
        document.getElementById('filename').value = 'main.py';
        this.resetFileSelector();
        this.hideResults();
    }

    resetFileSelector() {
        // Reset to manual filename input
        const filenameInput = document.getElementById('filename');
        const fileSelect = document.getElementById('repoFileSelect');
        const fileSourceInfo = document.getElementById('fileSourceInfo');
        
        filenameInput.classList.remove('hidden');
        fileSelect.classList.add('hidden');
        fileSourceInfo.classList.add('hidden');
        fileSelect.value = '';
    }

    showStatus(message) {
        document.getElementById('statusText').textContent = message;
        document.getElementById('statusPanel').classList.remove('hidden');
        this.hideResults();
    }

    hideStatus() {
        document.getElementById('statusPanel').classList.add('hidden');
    }

    hideResults() {
        document.getElementById('resultsSection').classList.add('hidden');
        document.getElementById('summaryPanel').classList.add('hidden');
        document.getElementById('frameworkPanel').classList.add('hidden');
    }

    showResults() {
        document.getElementById('resultsSection').classList.remove('hidden');
    }

    async auditCode() {
        const code = document.getElementById('codeInput').value.trim();
        const filename = document.getElementById('filename').value || 'main.py';
        
        if (!code) {
            alert('Please enter some code to analyze');
            return;
        }

        this.showStatus('Running static analysis...');

        try {
            const endpoint = document.getElementById('filterFalsePositives').checked ? '/audit' : '/audit/no-filter';
            const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    files: [{
                        filename: filename,
                        content: code
                    }],
                    options: {
                        level: document.getElementById('analysisLevel').value,
                        framework: 'auto',
                        target: 'gpu'
                    }
                })
            });

            if (!response.ok) {
                throw new Error(`Analysis failed: ${response.statusText}`);
            }

            const data = await response.json();
            this.currentResults = data;
            this.displayResults(data);
            
        } catch (error) {
            console.error('Audit error:', error);
            alert(`Analysis failed: ${error.message}`);
        } finally {
            this.hideStatus();
        }
    }

    async improveCode() {
        const code = document.getElementById('codeInput').value.trim();
        const filename = document.getElementById('filename').value || 'main.py';
        const apiKey = document.getElementById('apiKey').value.trim();
        const aiProvider = document.getElementById('aiProvider').value;
        
        if (!code) {
            alert('Please enter some code to improve');
            return;
        }

        if (!apiKey) {
            alert('Please enter your AI API key to use improvement features');
            return;
        }

        this.showStatus('AI is improving your code...');

        try {
            // First audit the code to get issues
            const auditResponse = await fetch(`${this.apiBaseUrl}/audit`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    files: [{
                        filename: filename,
                        content: code
                    }],
                    options: {
                        level: document.getElementById('analysisLevel').value,
                        framework: 'auto',
                        target: 'gpu'
                    }
                })
            });

            if (!auditResponse.ok) {
                throw new Error(`Initial analysis failed: ${auditResponse.statusText}`);
            }

            const auditData = await auditResponse.json();

            // Then improve the code using AI
            const improvePayload = {
                original_code: code,
                filename: filename,
                issues: auditData.issues || [],
                fixes: auditData.fixes || [],
                improvement_level: 'moderate',
                preserve_functionality: true,
                ai_provider: aiProvider,
                ai_api_key: apiKey
            };

            // Add repository context if available
            if (this.repositoryContext) {
                improvePayload.github_repo_url = this.repositoryContext.url;
                if (this.repositoryContext.token) {
                    improvePayload.github_token = this.repositoryContext.token;
                }
            }

            const endpoint = this.repositoryContext ? '/improve/with-repo-context' : '/improve/code';
            const improveResponse = await fetch(`${this.apiBaseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(improvePayload)
            });

            if (!improveResponse.ok) {
                throw new Error(`Code improvement failed: ${improveResponse.statusText}`);
            }

            const improveData = await improveResponse.json();
            
            // Combine audit and improvement results
            this.currentResults = {
                ...auditData,
                improved_code: improveData.improved_code,
                applied_fixes: improveData.applied_fixes,
                improvement_summary: improveData.improvement_summary,
                confidence_score: improveData.confidence_score,
                warnings: improveData.warnings,
                repository_context_used: this.repositoryContext && improveData.repository_context_used
            };
            
            this.displayResults(this.currentResults, true);
            
            // Show context enhancement notice if repository context was used
            if (this.repositoryContext && improveData.repository_context_used) {
                this.showContextEnhancementNotice(improveData);
            }
            
        } catch (error) {
            console.error('Improvement error:', error);
            alert(`Code improvement failed: ${error.message}`);
        } finally {
            this.hideStatus();
        }
    }

    async auditAndImprove() {
        const code = document.getElementById('codeInput').value.trim();
        const filename = document.getElementById('filename').value || 'main.py';
        const apiKey = document.getElementById('apiKey').value.trim();
        const aiProvider = document.getElementById('aiProvider').value;
        
        if (!code) {
            alert('Please enter some code to analyze');
            return;
        }

        if (!apiKey) {
            alert('Please enter your AI API key to use improvement features');
            return;
        }

        this.showStatus('Running comprehensive analysis and AI improvement...');

        try {
            const payload = {
                files: [{
                    filename: filename,
                    content: code
                }],
                options: {
                    level: document.getElementById('analysisLevel').value,
                    framework: 'auto',
                    target: 'gpu'
                },
                ai_provider: aiProvider,
                ai_api_key: apiKey
            };

            // Add repository context if available
            if (this.repositoryContext) {
                payload.github_repo_url = this.repositoryContext.url;
                if (this.repositoryContext.token) {
                    payload.github_token = this.repositoryContext.token;
                }
            }

            const response = await fetch(`${this.apiBaseUrl}/audit-and-improve`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`Analysis failed: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Transform audit-and-improve response to match displayResults format
            const transformedData = {
                issues: data.audit_results?.issues || [],
                fixes: data.audit_results?.fixes || [],
                summary: data.audit_results?.summary || '',
                frameworks: ['pytorch'], // Default, could be detected from options
                confidence_score: data.combined_summary?.average_ai_confidence || 0,
                // Get improved code from AI improvements if available
                improved_code: data.ai_improvements && Object.keys(data.ai_improvements).length > 0 
                    ? Object.values(data.ai_improvements)[0]?.improved_code 
                    : null,
                applied_fixes: data.ai_improvements && Object.keys(data.ai_improvements).length > 0 
                    ? Object.values(data.ai_improvements)[0]?.applied_fixes || []
                    : [],
                improvement_summary: data.ai_improvements && Object.keys(data.ai_improvements).length > 0 
                    ? Object.values(data.ai_improvements)[0]?.improvement_summary 
                    : 'AI improvement not available (API key required)',
                warnings: data.ai_improvements && Object.keys(data.ai_improvements).length > 0 
                    ? Object.values(data.ai_improvements)[0]?.warnings || []
                    : [],
                repository_context_used: this.repositoryContext && data.ai_improvements && 
                    Object.values(data.ai_improvements)[0]?.repository_context_used
            };
            
            this.currentResults = transformedData;
            this.displayResults(transformedData, true);
            
            // Show context enhancement notice if repository context was used
            if (this.repositoryContext && transformedData.repository_context_used) {
                this.showContextEnhancementNotice(data.ai_improvements ? Object.values(data.ai_improvements)[0] : {});
            }
            
        } catch (error) {
            console.error('Audit and improve error:', error);
            alert(`Analysis failed: ${error.message}`);
        } finally {
            this.hideStatus();
        }
    }

    displayResults(data, hasImprovedCode = false) {
        try {
            // Ensure data is valid
            if (!data || typeof data !== 'object') {
                console.warn('Invalid data provided to displayResults:', data);
                return;
            }

            // Update summary with null safety
            const issueCountEl = document.getElementById('issueCount');
            const fixCountEl = document.getElementById('fixCount');
            const summaryPanelEl = document.getElementById('summaryPanel');
            
            if (issueCountEl) issueCountEl.textContent = data.issues?.length || 0;
            if (fixCountEl) fixCountEl.textContent = data.fixes?.length || 0;
            if (summaryPanelEl) summaryPanelEl.classList.remove('hidden');

            // Show confidence score if available
            if (data.confidence_score !== undefined) {
                const confidenceEl = document.getElementById('confidence');
                const confidenceScoreEl = document.getElementById('confidenceScore');
                if (confidenceEl) confidenceEl.textContent = Math.round(data.confidence_score * 100);
                if (confidenceScoreEl) confidenceScoreEl.classList.remove('hidden');
            }
        } catch (error) {
            console.error('Error in displayResults:', error);
        }

        // Display frameworks with error handling
        if (data.frameworks && Array.isArray(data.frameworks) && data.frameworks.length > 0) {
            const frameworkList = document.getElementById('frameworkList');
            const frameworkPanel = document.getElementById('frameworkPanel');
            
            if (frameworkList && frameworkPanel) {
                frameworkList.innerHTML = '';
                data.frameworks.forEach(framework => {
                    if (framework) {
                        const badge = document.createElement('span');
                        badge.className = 'px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full';
                        badge.textContent = String(framework);
                        frameworkList.appendChild(badge);
                    }
                });
                frameworkPanel.classList.remove('hidden');
            }
        }

        // Display issues
        this.displayIssues(data.issues || [], data.fixes || []);

        // Display improved code if available with error handling
        if (hasImprovedCode && data.improved_code) {
            const improvedCodeEl = document.getElementById('improvedCode');
            if (improvedCodeEl) {
                improvedCodeEl.textContent = data.improved_code;
                
                // Defer syntax highlighting to avoid blocking UI
                if (window.Prism && window.Prism.highlightElement) {
                    if (window.requestIdleCallback) {
                        requestIdleCallback(() => {
                            try {
                                Prism.highlightElement(improvedCodeEl);
                            } catch (error) {
                                console.warn('Syntax highlighting failed:', error);
                            }
                        });
                    } else {
                        setTimeout(() => {
                            try {
                                Prism.highlightElement(improvedCodeEl);
                            } catch (error) {
                                console.warn('Syntax highlighting failed:', error);
                            }
                        }, 16);
                    }
                }
            }
        }

        // Generate comprehensive report (debounced)
        this.debouncedGenerateReport(data);

        this.showResults();
    }

    displayIssues(issues, fixes) {
        const issuesList = document.getElementById('issuesList');
        
        if (!issuesList) {
            console.error('Issues list element not found');
            return;
        }

        if (!Array.isArray(issues) || issues.length === 0) {
            issuesList.innerHTML = '<p class="text-green-600 font-medium">No issues found! Your code looks good.</p>';
            return;
        }

        // Use DocumentFragment for batched DOM operations
        const fragment = document.createDocumentFragment();
        
        // Pre-filter fixes by line/filename for better performance
        const fixesMap = new Map();
        fixes.forEach(fix => {
            const key = `${fix.filename}:${fix.line}`;
            if (!fixesMap.has(key)) {
                fixesMap.set(key, []);
            }
            fixesMap.get(key).push(fix);
        });

        issues.forEach((issue, index) => {
            const issueDiv = this.createIssueElement(issue, fixesMap.get(`${issue.filename}:${issue.line}`) || []);
            fragment.appendChild(issueDiv);
        });

        // Single DOM update
        issuesList.innerHTML = '';
        issuesList.appendChild(fragment);
    }

    createIssueElement(issue, relatedFixes) {
        const issueDiv = document.createElement('div');
        issueDiv.className = `mb-4 p-4 rounded-lg issue-severity-${issue.severity || 'low'}`;

        // Build HTML string for better performance
        const severityClass = this.getSeverityBadgeClass(issue.severity);
        const columnInfo = issue.column ? ` <strong>Column:</strong> ${issue.column}` : '';
        const fixesHtml = relatedFixes.length > 0 ? this.renderFixes(relatedFixes) : '';

        issueDiv.innerHTML = `
            <div class="flex justify-between items-start mb-2">
                <h4 class="font-semibold text-gray-800">${this.escapeHtml(issue.type || 'Code Issue')}</h4>
                <div class="flex space-x-2">
                    <span class="px-2 py-1 text-xs rounded ${severityClass}">${this.escapeHtml(issue.severity || 'low')}</span>
                    <span class="px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded">${this.escapeHtml(issue.source || 'unknown')}</span>
                </div>
            </div>
            <p class="text-gray-700 mb-2">${this.escapeHtml(issue.description)}</p>
            <div class="text-sm text-gray-600 mb-3">
                <strong>File:</strong> ${this.escapeHtml(issue.filename)} 
                <strong>Line:</strong> ${issue.line}${columnInfo}
            </div>
            ${fixesHtml}
        `;

        return issueDiv;
    }

    renderFixes(fixes) {
        if (fixes.length === 0) return '';

        const fixesHtml = fixes.map(fix => `
            <div class="bg-green-50 p-3 rounded border-l-4 border-green-400 mt-2">
                <h5 class="font-medium text-green-800 mb-1">Suggested Fix:</h5>
                <p class="text-green-700 text-sm mb-2">${this.escapeHtml(fix.suggestion || fix.description || 'No description available')}</p>
                ${fix.replacement_code ? `
                    <div class="bg-green-100 p-2 rounded text-sm">
                        <strong>Replace with:</strong>
                        <pre class="mt-1 font-mono text-xs">${this.escapeHtml(fix.replacement_code)}</pre>
                    </div>
                ` : ''}
                ${fix.diff ? `
                    <details class="mt-2">
                        <summary class="cursor-pointer text-green-600 text-sm">View Diff</summary>
                        <pre class="mt-1 text-xs bg-gray-100 p-2 rounded">${this.escapeHtml(fix.diff)}</pre>
                    </details>
                ` : ''}
            </div>
        `).join('');

        return `<div class="fixes-section">${fixesHtml}</div>`;
    }

    getSeverityBadgeClass(severity) {
        switch (severity) {
            case 'high': return 'bg-red-100 text-red-800';
            case 'medium': return 'bg-yellow-100 text-yellow-800';
            default: return 'bg-blue-100 text-blue-800';
        }
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.tab === tabName) {
                btn.classList.add('active');
            }
        });

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.add('hidden');
        });
        document.getElementById(`${tabName}Tab`).classList.remove('hidden');

        // Defer syntax highlighting using requestIdleCallback for better performance
        if (tabName === 'improved' && this.currentResults?.improved_code) {
            if (window.requestIdleCallback) {
                requestIdleCallback(() => {
                    Prism.highlightElement(document.getElementById('improvedCode'));
                });
            } else {
                // Fallback for older browsers
                setTimeout(() => {
                    Prism.highlightElement(document.getElementById('improvedCode'));
                }, 16); // ~1 frame delay
            }
        }
    }

    async generateReport(data) {
        try {
            const code = document.getElementById('codeInput').value;
            const filename = document.getElementById('filename').value || 'main.py';
            const filtering = document.getElementById('filterFalsePositives').checked;
            
            // Create cache key
            const cacheKey = `${filename}:${code.length}:${filtering}:${Date.now() - (Date.now() % 300000)}`; // 5min cache
            
            // Check cache first
            if (this.reportCache.has(cacheKey)) {
                document.getElementById('reportContent').innerHTML = this.reportCache.get(cacheKey);
                return;
            }

            const response = await fetch(`${this.apiBaseUrl}/reports/improvement-analysis`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    files: [{
                        filename: filename,
                        content: code
                    }],
                    format: 'html',
                    include_ai_suggestions: true,
                    apply_false_positive_filtering: filtering
                })
            });

            if (response.ok) {
                const reportData = await response.json();
                const reportContent = reportData.report || 'Report generation failed';
                
                // Cache the result
                this.reportCache.set(cacheKey, reportContent);
                
                // Clean cache if it gets too large
                if (this.reportCache.size > 10) {
                    const firstKey = this.reportCache.keys().next().value;
                    this.reportCache.delete(firstKey);
                }
                
                document.getElementById('reportContent').innerHTML = reportContent;
            }
        } catch (error) {
            console.error('Report generation error:', error);
            document.getElementById('reportContent').innerHTML = 'Failed to generate comprehensive report';
        }
    }

    // Utility function for debouncing
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    copyImprovedCode() {
        const improvedCode = document.getElementById('improvedCode').textContent;
        if (improvedCode) {
            navigator.clipboard.writeText(improvedCode).then(() => {
                alert('Improved code copied to clipboard!');
            });
        }
    }

    downloadImprovedCode() {
        const improvedCode = document.getElementById('improvedCode').textContent;
        const filename = document.getElementById('filename').value || 'improved_main.py';
        
        if (improvedCode) {
            const blob = new Blob([improvedCode], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename.replace('.py', '_improved.py');
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    }

    async exportReport(format) {
        if (!this.currentResults) {
            alert('Please run an analysis first');
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/reports/improvement-analysis`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    files: [{
                        filename: document.getElementById('filename').value || 'main.py',
                        content: document.getElementById('codeInput').value
                    }],
                    format: format,
                    include_ai_suggestions: true,
                    apply_false_positive_filtering: document.getElementById('filterFalsePositives').checked
                })
            });

            if (response.ok) {
                const data = await response.json();
                const content = data.report;
                const mimeType = format === 'html' ? 'text/html' : 'text/markdown';
                const extension = format === 'html' ? 'html' : 'md';
                
                const blob = new Blob([content], { type: mimeType });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `codeguard_report.${extension}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
        } catch (error) {
            console.error('Export error:', error);
            alert('Failed to export report');
        }
    }

    escapeHtml(unsafe) {
        if (unsafe === null || unsafe === undefined) {
            return '';
        }
        return String(unsafe)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // FIM Completion Methods
    openFimTab() {
        this.switchTab('fim');
        document.getElementById('resultsSection').classList.remove('hidden');
    }

    loadFimExample() {
        const examplePrefix = `def secure_model_loader(model_path: str):
    """Load ML model with security checks."""
    import torch
    import os
    
    # TODO: Add proper validation and security checks`;
        
        const exampleSuffix = `    
    model = torch.load(model_path)
    return model`;
        
        document.getElementById('fimPrefix').value = examplePrefix;
        document.getElementById('fimSuffix').value = exampleSuffix;
    }

    async runFimCompletion() {
        const prefix = document.getElementById('fimPrefix').value.trim();
        const suffix = document.getElementById('fimSuffix').value.trim();
        const apiKey = document.getElementById('apiKey').value.trim();
        const provider = document.getElementById('aiProvider').value;
        
        if (!prefix) {
            alert('Please enter a code prefix');
            return;
        }
        
        if (provider === 'deepseek' && !apiKey) {
            alert('DeepSeek API key is required for FIM completion');
            return;
        }
        
        this.showStatus('Running FIM completion...');
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/improve/fim-completion`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prefix: prefix,
                    suffix: suffix,
                    ai_provider: provider,
                    ai_api_key: apiKey,
                    max_tokens: 2000
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            this.displayFimResults(result);
            this.hideStatus();
            
        } catch (error) {
            this.hideStatus();
            console.error('FIM completion error:', error);
            alert(`FIM completion failed: ${error.message}`);
        }
    }

    displayFimResults(result) {
        const fimResult = document.getElementById('fimResult');
        const completedCode = result.prefix + '\n' + (result.completion || '') + '\n' + result.suffix;
        fimResult.textContent = completedCode;
        
        if (result.confidence_score !== undefined) {
            document.getElementById('fimConfidenceScore').textContent = Math.round(result.confidence_score * 100);
            document.getElementById('fimConfidence').classList.remove('hidden');
        }
        
        this.currentFimResult = result;
        document.getElementById('resultsSection').classList.remove('hidden');
        this.switchTab('fim');
    }

    copyFimResult() {
        if (!this.currentFimResult) return;
        
        const completedCode = this.currentFimResult.prefix + '\n' + 
                             (this.currentFimResult.completion || '') + '\n' + 
                             this.currentFimResult.suffix;
        
        navigator.clipboard.writeText(completedCode).then(() => {
            const btn = document.getElementById('copyFimResult');
            const originalText = btn.textContent;
            btn.textContent = 'Copied!';
            setTimeout(() => btn.textContent = originalText, 2000);
        });
    }

    applyFimResult() {
        if (!this.currentFimResult) return;
        
        const completedCode = this.currentFimResult.prefix + '\n' + 
                             (this.currentFimResult.completion || '') + '\n' + 
                             this.currentFimResult.suffix;
        
        document.getElementById('codeInput').value = completedCode;
        
        const btn = document.getElementById('applyFimResult');
        const originalText = btn.textContent;
        btn.textContent = 'Applied!';
        setTimeout(() => btn.textContent = originalText, 2000);
    }

    // Repository Context Methods
    validateRepoUrl() {
        const repoUrl = document.getElementById('githubRepoUrl').value.trim();
        const analyzeBtn = document.getElementById('analyzeRepo');
        
        if (repoUrl && this.isValidGitHubUrl(repoUrl)) {
            analyzeBtn.disabled = false;
            analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        } else {
            analyzeBtn.disabled = true;
            analyzeBtn.classList.add('opacity-50', 'cursor-not-allowed');
        }
    }

    isValidGitHubUrl(url) {
        const githubPattern = /^https:\/\/github\.com\/[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+\/?$/;
        return githubPattern.test(url);
    }

    async analyzeRepository() {
        const repoUrl = document.getElementById('githubRepoUrl').value.trim();
        const githubToken = document.getElementById('githubToken').value.trim();
        const statusEl = document.getElementById('repoStatus');
        const repoInfoEl = document.getElementById('repoInfo');
        const analyzeBtn = document.getElementById('analyzeRepo');

        if (!repoUrl) return;

        statusEl.innerHTML = '<span class="text-blue-600">Analyzing repository...</span>';
        analyzeBtn.disabled = true;
        
        try {
            const payload = { repo_url: repoUrl };
            if (githubToken) {
                payload.github_token = githubToken;
            }

            const response = await fetch(`${this.apiBaseUrl}/repo/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (response.ok && result.status === 'success') {
                statusEl.innerHTML = '<span class="text-green-600">✓ Repository analyzed successfully</span>';
                this.displayRepositoryInfo(result);
                
                this.repositoryContext = {
                    url: repoUrl,
                    token: githubToken,
                    info: result.repository,
                    contextSummary: result.context_summary
                };
                
                // Fetch repository files for dropdown selection
                await this.fetchRepositoryFiles();
                
                // Show Smart Context Improve button
                this.showSmartContextButton();
                
            } else {
                statusEl.innerHTML = '<span class="text-red-600">✗ Analysis failed: ' + (result.error || result.detail || 'Unknown error') + '</span>';
                repoInfoEl.classList.add('hidden');
            }

        } catch (error) {
            statusEl.innerHTML = '<span class="text-red-600">✗ Error: ' + error.message + '</span>';
            repoInfoEl.classList.add('hidden');
        } finally {
            analyzeBtn.disabled = false;
        }
    }

    displayRepositoryInfo(result) {
        const repoInfoEl = document.getElementById('repoInfo');
        const repoDetailsEl = document.getElementById('repoDetails');
        const repo = result.repository;

        repoDetailsEl.innerHTML = `
            <div><strong>Repository:</strong> ${repo.owner}/${repo.name}</div>
            <div><strong>Language:</strong> ${repo.language}</div>
            <div><strong>Framework:</strong> ${repo.framework}</div>
            <div><strong>Dependencies:</strong> ${repo.dependencies_count} packages</div>
            ${repo.description ? `<div><strong>Description:</strong> ${repo.description}</div>` : ''}
            ${repo.topics && repo.topics.length > 0 ? `<div><strong>Topics:</strong> ${repo.topics.join(', ')}</div>` : ''}
            <div class="mt-2 text-xs">
                <strong>Key Dependencies:</strong> ${repo.key_dependencies.slice(0, 5).join(', ')}
                ${repo.key_dependencies.length > 5 ? ` and ${repo.key_dependencies.length - 5} more...` : ''}
            </div>
        `;

        repoInfoEl.classList.remove('hidden');
    }

    async fetchRepositoryFiles() {
        if (!this.repositoryContext) return;

        try {
            const payload = { 
                repo_url: this.repositoryContext.url,
                max_files: 50
            };
            if (this.repositoryContext.token) {
                payload.github_token = this.repositoryContext.token;
            }

            const response = await fetch(`${this.apiBaseUrl}/repo/files`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (response.ok && result.status === 'success' && result.files.length > 0) {
                this.populateFileDropdown(result.files);
                this.showFileSelector();
                this.showSmartContextButton();
            }

        } catch (error) {
            console.error('Error fetching repository files:', error);
        }
    }

    populateFileDropdown(files) {
        const selectEl = document.getElementById('repoFileSelect');
        const filenameInput = document.getElementById('filename');
        
        // Clear existing options except the first one
        selectEl.innerHTML = '<option value="">Select a file from repository...</option>';
        
        // Add files to dropdown
        files.forEach(file => {
            const option = document.createElement('option');
            option.value = file.path;
            option.textContent = file.directory ? 
                `${file.directory}/${file.filename}` : 
                file.filename;
            selectEl.appendChild(option);
        });
        
        // Store files data for later use
        this.repositoryFiles = files;
    }

    showFileSelector() {
        const filenameInput = document.getElementById('filename');
        const fileSelect = document.getElementById('repoFileSelect');
        
        // Hide filename input and show dropdown
        filenameInput.classList.add('hidden');
        fileSelect.classList.remove('hidden');
    }

    async loadSelectedRepoFile() {
        const selectEl = document.getElementById('repoFileSelect');
        const selectedPath = selectEl.value;
        
        if (!selectedPath || !this.repositoryContext) return;

        const statusEl = document.getElementById('repoStatus');
        statusEl.innerHTML = '<span class="text-blue-600">Loading file...</span>';

        try {
            const payload = {
                repo_url: this.repositoryContext.url,
                file_path: selectedPath
            };
            if (this.repositoryContext.token) {
                payload.github_token = this.repositoryContext.token;
            }

            const response = await fetch(`${this.apiBaseUrl}/repo/file-content`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (response.ok && result.status === 'success') {
                // Load file content into editor
                document.getElementById('codeInput').value = result.file.content;
                document.getElementById('filename').value = result.file.filename;
                
                // Show file source info
                document.getElementById('fileSourceInfo').classList.remove('hidden');
                
                statusEl.innerHTML = '<span class="text-green-600">✓ File loaded successfully</span>';
            } else {
                statusEl.innerHTML = '<span class="text-red-600">✗ Failed to load file</span>';
            }

        } catch (error) {
            console.error('Error loading file:', error);
            statusEl.innerHTML = '<span class="text-red-600">✗ Error loading file</span>';
        }
    }

    showSmartContextButton() {
        const contextBtn = document.getElementById('improveWithContextBtn');
        contextBtn.classList.remove('hidden');
    }

    async improveWithRelatedContext() {
        const code = document.getElementById('codeInput').value.trim();
        const filename = document.getElementById('filename').value || 'main.py';
        const apiKey = document.getElementById('apiKey').value.trim();
        const aiProvider = document.getElementById('aiProvider').value;
        
        if (!code) {
            alert('Please enter some code to improve');
            return;
        }

        if (!apiKey) {
            alert('Please enter your AI API key to use improvement features');
            return;
        }

        if (!this.repositoryContext) {
            alert('Please analyze a GitHub repository first to enable Smart Context Improve');
            return;
        }

        // Get the selected file path
        const selectEl = document.getElementById('repoFileSelect');
        const selectedPath = selectEl.value;
        
        if (!selectedPath) {
            alert('Please select a file from the repository dropdown');
            return;
        }

        this.showStatus('Discovering related files and improving with AI context...');

        try {
            const payload = {
                original_code: code,
                filename: filename,
                github_repo_url: this.repositoryContext.url,
                github_token: this.repositoryContext.token,
                target_file_path: selectedPath,
                ai_provider: aiProvider,
                ai_api_key: apiKey,
                improvement_level: 'moderate',
                max_related_files: 5
            };

            const response = await fetch(`${this.apiBaseUrl}/improve/with-related-context`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`Smart Context Improve failed: ${response.statusText}`);
            }

            const result = await response.json();
            
            // Display results with context information
            this.displaySmartContextResults(result);
            
        } catch (error) {
            console.error('Smart Context Improve error:', error);
            alert(`Smart Context Improve failed: ${error.message}`);
        } finally {
            this.hideStatus();
        }
    }

    displaySmartContextResults(result) {
        // Show the improved code
        document.getElementById('improvedCode').textContent = result.improved_code;
        document.getElementById('confidence').textContent = Math.round(result.confidence_score * 100);
        document.getElementById('confidenceScore').classList.remove('hidden');

        // Show context enhancement notification
        const enhancement = document.createElement('div');
        enhancement.className = 'bg-indigo-50 border border-indigo-200 rounded-lg p-4 mb-4';
        enhancement.innerHTML = `
            <h4 class="font-semibold text-indigo-800 mb-2">🔗 Smart Context Enhancement</h4>
            <p class="text-indigo-700 text-sm mb-2">
                AI analysis enhanced with ${result.related_files_used} related files from repository
            </p>
            <div class="text-xs text-indigo-600">
                <strong>Related files used:</strong>
                ${result.related_files.map(f => `${f.filename} (${f.reason})`).join(', ')}
            </div>
        `;

        const improvedTab = document.getElementById('improvedTab');
        improvedTab.insertBefore(enhancement, improvedTab.firstChild);

        // Switch to improved tab and show results
        this.switchTab('improved');
        document.getElementById('resultsSection').classList.remove('hidden');
        document.getElementById('summaryPanel').classList.remove('hidden');

        // Update summary with context info
        document.getElementById('issueCount').textContent = result.applied_fixes.length;
        document.getElementById('fixCount').textContent = result.applied_fixes.length;
    }

    showContextEnhancementNotice(result) {
        const resultsEl = document.getElementById('resultsContent');
        const contextNotice = document.createElement('div');
        contextNotice.className = 'bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4';
        contextNotice.innerHTML = `
            <div class="flex items-center">
                <svg class="w-5 h-5 text-blue-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                <span class="font-medium text-blue-800">Repository Context Enhanced</span>
            </div>
            <p class="text-blue-700 text-sm mt-1">
                AI suggestions improved using ${this.repositoryContext.info.framework} patterns and project-specific context
            </p>
            <div class="text-xs text-blue-600 mt-2">
                Context-aware improvements applied with enhanced accuracy
            </div>
        `;
        resultsEl.insertBefore(contextNotice, resultsEl.firstChild);
    }

    // System Management Methods
    async showCacheStats() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/cache/stats`);
            const stats = await response.json();
            
            const message = `Analysis Cache Statistics:
• Cache Entries: ${stats.file_cache.entries}
• Valid Entries: ${stats.file_cache.valid_entries}
• Storage Used: ${stats.file_cache.size_mb} MB
• TTL Hours: ${stats.file_cache.ttl_hours}
• Performance Impact: ${stats.performance_impact.estimated_speedup}`;

            alert(message);
        } catch (error) {
            console.error('Error fetching cache stats:', error);
            alert('Failed to load cache statistics');
        }
    }

    async clearCache() {
        if (!confirm('Are you sure you want to clear the analysis cache? This will remove all cached results.')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/cache/clear`, {
                method: 'POST'
            });

            if (response.ok) {
                alert('Analysis cache cleared successfully');
            } else {
                throw new Error('Failed to clear cache');
            }
        } catch (error) {
            console.error('Error clearing cache:', error);
            alert('Failed to clear cache');
        }
    }

    async showRuleConfig() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/rules/config`);
            const config = await response.json();
            
            const message = `Rule Configuration:
• Total Rules: ${config.configuration.total_rules}
• Enabled Rules: ${config.configuration.enabled_rules}
• Disabled Rules: ${config.configuration.disabled_rules}

Rules by Severity:
${Object.entries(config.configuration.by_severity).map(([severity, count]) => `• ${severity}: ${count}`).join('\n')}

Available Rule Sets:
${Object.entries(config.rule_sets).map(([name, rules]) => `• ${name}: ${rules.length} rules`).join('\n')}`;

            alert(message);
        } catch (error) {
            console.error('Error fetching rule configuration:', error);
            alert('Failed to load rule configuration');
        }
    }

    async showSystemHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/system/health/detailed`);
            const health = await response.json();
            
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

            alert(message);
        } catch (error) {
            console.error('Error fetching system health:', error);
            alert('Failed to load system health');
        }
    }
}

// Initialize the playground when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new CodeGuardPlayground();
});