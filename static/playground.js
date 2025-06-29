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
        // API key management
        document.getElementById('rememberKey').addEventListener('change', (e) => {
            if (e.target.checked) {
                const apiKey = document.getElementById('apiKey').value;
                const provider = document.getElementById('aiProvider').value;
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

        document.getElementById('apiKey').addEventListener('input', (e) => {
            if (document.getElementById('rememberKey').checked) {
                localStorage.setItem('codeguard_api_key', e.target.value);
            }
        });

        document.getElementById('aiProvider').addEventListener('change', (e) => {
            if (document.getElementById('rememberKey').checked) {
                localStorage.setItem('codeguard_ai_provider', e.target.value);
            }
        });

        // Code management
        document.getElementById('loadExample').addEventListener('click', () => this.loadExampleCode());
        document.getElementById('clearCode').addEventListener('click', () => this.clearCode());

        // Analysis buttons
        document.getElementById('auditBtn').addEventListener('click', () => this.auditCode());
        document.getElementById('improveBtn').addEventListener('click', () => this.improveCode());
        document.getElementById('auditImproveBtn').addEventListener('click', () => this.auditAndImprove());
        document.getElementById('fimBtn').addEventListener('click', () => this.openFimTab());
        
        // FIM completion buttons
        document.getElementById('runFimBtn').addEventListener('click', () => this.runFimCompletion());
        document.getElementById('loadFimExample').addEventListener('click', () => this.loadFimExample());
        document.getElementById('copyFimResult').addEventListener('click', () => this.copyFimResult());
        document.getElementById('applyFimResult').addEventListener('click', () => this.applyFimResult());

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // Export buttons
        document.getElementById('copyImproved').addEventListener('click', () => this.copyImprovedCode());
        document.getElementById('downloadImproved').addEventListener('click', () => this.downloadImprovedCode());
        document.getElementById('exportMarkdown').addEventListener('click', () => this.exportReport('markdown'));
        document.getElementById('exportHtml').addEventListener('click', () => this.exportReport('html'));
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
        this.hideResults();
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
            const improveResponse = await fetch(`${this.apiBaseUrl}/improve/code`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    original_code: code,
                    filename: filename,
                    issues: auditData.issues || [],
                    fixes: auditData.fixes || [],
                    improvement_level: 'moderate',
                    preserve_functionality: true,
                    ai_provider: aiProvider,
                    ai_api_key: apiKey
                })
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
                warnings: improveData.warnings
            };
            
            this.displayResults(this.currentResults, true);
            
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
            const response = await fetch(`${this.apiBaseUrl}/audit-and-improve`, {
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
                    },
                    ai_provider: aiProvider,
                    ai_api_key: apiKey
                })
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
                    : []
            };
            
            this.currentResults = transformedData;
            this.displayResults(transformedData, true);
            
        } catch (error) {
            console.error('Audit and improve error:', error);
            alert(`Analysis failed: ${error.message}`);
        } finally {
            this.hideStatus();
        }
    }

    displayResults(data, hasImprovedCode = false) {
        // Update summary
        document.getElementById('issueCount').textContent = data.issues?.length || 0;
        document.getElementById('fixCount').textContent = data.fixes?.length || 0;
        document.getElementById('summaryPanel').classList.remove('hidden');

        // Show confidence score if available
        if (data.confidence_score !== undefined) {
            document.getElementById('confidence').textContent = Math.round(data.confidence_score * 100);
            document.getElementById('confidenceScore').classList.remove('hidden');
        }

        // Display frameworks
        if (data.frameworks && data.frameworks.length > 0) {
            const frameworkList = document.getElementById('frameworkList');
            frameworkList.innerHTML = '';
            data.frameworks.forEach(framework => {
                const badge = document.createElement('span');
                badge.className = 'px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full';
                badge.textContent = framework;
                frameworkList.appendChild(badge);
            });
            document.getElementById('frameworkPanel').classList.remove('hidden');
        }

        // Display issues
        this.displayIssues(data.issues || [], data.fixes || []);

        // Display improved code if available
        if (hasImprovedCode && data.improved_code) {
            document.getElementById('improvedCode').textContent = data.improved_code;
            
            // Defer syntax highlighting to avoid blocking UI
            if (window.requestIdleCallback) {
                requestIdleCallback(() => {
                    Prism.highlightElement(document.getElementById('improvedCode'));
                });
            } else {
                setTimeout(() => {
                    Prism.highlightElement(document.getElementById('improvedCode'));
                }, 16);
            }
        }

        // Generate comprehensive report (debounced)
        this.debouncedGenerateReport(data);

        this.showResults();
    }

    displayIssues(issues, fixes) {
        const issuesList = document.getElementById('issuesList');

        if (issues.length === 0) {
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
}

// Initialize the playground when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new CodeGuardPlayground();
});