from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from models import AuditRequest, AuditResponse
from audit import analyze_code
from auth import verify_api_key, get_current_user
from rule_loader import CustomRuleEngine
from telemetry import telemetry_collector, metrics_analyzer
from dashboard import get_dashboard
from historical_timeline import get_timeline_generator
from gpt_connector import get_gpt_connector, get_issue_explainer
from project_templates import MLProjectGenerator
from chatgpt_integration import get_code_improver, get_batch_improver, CodeImprovementRequest
from multi_ai_integration import get_multi_ai_manager
from ml_performance_heatmap import heatmap_api, HeatmapConfig
import uuid
import time

# Create FastAPI app
app = FastAPI(
    title="CodeGuard API",
    version="1.0.0",
    description="Audits ML and RL Python files for issues using static analysis tools.",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize project generator
project_generator = MLProjectGenerator()

# Mount static files for the playground
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Root endpoint providing basic service information and health check."""
    return {
        "status": "healthy",
        "service": "CodeGuard API",
        "version": "1.0.0",
        "description": "Static code analysis service for ML/RL Python code",
        "endpoints": {
            "playground": "/playground",
            "audit": "/audit",
            "improve_code": "/improve/code",
            "improve_project": "/improve/project", 
            "audit_and_improve": "/audit-and-improve",
            "openapi": "/.well-known/openapi.yaml",
            "deployment_openapi": "/openapi-deployment.yaml",
            "docs": "/docs",
            "privacy": "/privacy-policy",
            "terms": "/terms-of-service"
        }
    }

@app.get("/playground", response_class=HTMLResponse)
async def playground():
    """Serve the CodeGuard Playground web interface."""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Playground not found")

@app.post("/audit", response_model=AuditResponse)
async def audit_code(request: AuditRequest, current_user: dict = Depends(get_current_user)):
    """
    Standard audit endpoint - includes ChatGPT false positive filtering by default.
    """
    return await _perform_audit(request, validate_with_ai=True)

@app.post("/audit/no-filter", response_model=AuditResponse)
async def audit_code_no_filter(request: AuditRequest, current_user: dict = Depends(get_current_user)):
    """
    Audit endpoint without ChatGPT false positive filtering for debugging.
    """
    return await _perform_audit(request, validate_with_ai=False)

async def _perform_audit(request: AuditRequest, validate_with_ai: bool = True):
    """
    Analyzes submitted Python code files and returns a report of issues and suggestions.
    
    Args:
        request: AuditRequest containing files to analyze
        validate_with_ai: Whether to apply ChatGPT false positive filtering
        
    Returns:
        AuditResponse with summary, issues, and fix suggestions
        
    Raises:
        HTTPException: If analysis fails
    """
    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Analyze request for telemetry
        request_analysis = metrics_analyzer.analyze_request(request)
        
        # Apply timeout logic for AI validation
        analysis_timeout = 40  # Total analysis timeout in seconds
        
        # Perform code analysis with configurable AI validation and timeout
        from enhanced_audit import EnhancedAuditEngine
        import asyncio
        
        engine = EnhancedAuditEngine(use_false_positive_filter=validate_with_ai)
        
        try:
            # Run analysis with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(engine.analyze_code, request),
                timeout=analysis_timeout
            )
        except asyncio.TimeoutError:
            # Fallback to analysis without AI filtering if timeout
            if validate_with_ai:
                print(f"AI validation timed out after {analysis_timeout}s, falling back to standard analysis")
                engine_fallback = EnhancedAuditEngine(use_false_positive_filter=False)
                response = engine_fallback.analyze_code(request)
                response.summary += " (AI validation timed out)"
            else:
                raise HTTPException(status_code=504, detail="Analysis timed out")
        
        # Calculate analysis time
        analysis_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Analyze response for telemetry
        response_analysis = metrics_analyzer.analyze_response(response)
        
        # Create and record telemetry session
        session = metrics_analyzer.create_session(
            session_id, request_analysis, response_analysis, analysis_time
        )
        telemetry_collector.record_audit_session(session)
        telemetry_collector.record_error_patterns(session_id, response.issues)
        
        # Record framework usage if detected
        if request_analysis['primary_framework']:
            telemetry_collector.record_framework_usage(request_analysis['primary_framework'])
        
        return response
        
    except Exception as e:
        # Record failed session for monitoring
        analysis_time = (time.time() - start_time) * 1000
        # Log error but don't expose internal details
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/.well-known/openapi.yaml")
async def get_openapi_spec():
    """Serve the OpenAPI specification for GPT Action integration."""
    openapi_path = ".well-known/openapi.yaml"
    if os.path.exists(openapi_path):
        return FileResponse(openapi_path, media_type="application/yaml")
    else:
        raise HTTPException(status_code=404, detail="OpenAPI specification not found")

@app.get("/openapi-deployment.yaml")
async def get_deployment_openapi_spec():
    """Serve the deployment OpenAPI specification for production use."""
    openapi_path = "openapi-deployment.yaml"
    if os.path.exists(openapi_path):
        return FileResponse(openapi_path, media_type="application/yaml")
    else:
        raise HTTPException(status_code=404, detail="Deployment OpenAPI specification not found")

@app.get("/auth/status")
async def auth_status(current_user: dict = Depends(get_current_user)):
    """Check authentication status and return user information."""
    return {
        "authenticated": True,
        "message": "API key is valid",
        "user_info": current_user
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "CodeGuard API"}

@app.get("/rules/summary")
async def get_rules_summary():
    """Get a summary of all loaded custom rules."""
    try:
        engine = CustomRuleEngine()
        return {
            "status": "success",
            "summary": engine.get_rule_summary()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rules summary: {str(e)}")

@app.post("/rules/reload")
async def reload_rules():
    """Reload all custom rules from rule files."""
    try:
        engine = CustomRuleEngine()
        engine.reload_rules()
        return {
            "status": "success",
            "message": f"Reloaded {engine.get_rule_count()} rules successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload rules: {str(e)}")

@app.get("/rules/by-tag/{tag}")
async def get_rules_by_tag(tag: str):
    """Get all rules that contain a specific tag."""
    try:
        engine = CustomRuleEngine()
        rules = engine.get_rules_by_tag(tag)
        return {
            "status": "success",
            "tag": tag,
            "count": len(rules),
            "rules": rules
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rules by tag: {str(e)}")

@app.get("/metrics/usage")
async def get_usage_metrics(days: int = 7):
    """Get usage metrics for the specified number of days."""
    try:
        metrics = telemetry_collector.get_usage_metrics(days)
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get usage metrics: {str(e)}")

@app.get("/metrics/frameworks")
async def get_framework_metrics():
    """Get framework usage statistics."""
    try:
        # Use in-memory data if database is unavailable
        if hasattr(telemetry_collector, 'memory_frameworks'):
            framework_stats = dict(telemetry_collector.memory_frameworks)
        else:
            framework_stats = {}
        
        return {
            "status": "success",
            "frameworks": framework_stats,
            "total_detections": sum(framework_stats.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get framework metrics: {str(e)}")

@app.get("/dashboard")
async def get_analytics_dashboard(days: int = 7):
    """Get comprehensive analytics dashboard data."""
    try:
        dashboard_instance = get_dashboard()
        data = dashboard_instance.generate_dashboard_data(days)
        return {
            "status": "success",
            "dashboard": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard: {str(e)}")

@app.get("/dashboard/export")
async def export_analytics_report(days: int = 7, format: str = "markdown"):
    """Export analytics report in various formats."""
    try:
        dashboard_instance = get_dashboard()
        data = dashboard_instance.generate_dashboard_data(days)
        report = dashboard_instance.export_report(data, format)
        
        if format.lower() == "markdown":
            return {"status": "success", "report": report, "format": "markdown"}
        else:
            return {"status": "success", "report": report, "format": format}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export report: {str(e)}")

@app.get("/timeline")
async def get_historical_timeline(days: int = 30, granularity: str = "daily"):
    """Get historical audit timeline showing trends over time."""
    try:
        timeline_gen = get_timeline_generator()
        timeline_data = timeline_gen.generate_timeline(days, granularity)
        return {
            "status": "success",
            "timeline": timeline_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate timeline: {str(e)}")

@app.get("/timeline/frameworks")
async def get_framework_trends(days: int = 30):
    """Get framework usage trends over time."""
    try:
        timeline_gen = get_timeline_generator()
        trends = timeline_gen.get_framework_trends(days)
        return {
            "status": "success",
            "trends": trends
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get framework trends: {str(e)}")

@app.post("/query/audits")
async def query_past_audits(request: dict):
    """Query past audits using natural language."""
    try:
        query = request.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")
        
        gpt_conn = get_gpt_connector()
        result = gpt_conn.query_audits(query)
        
        return {
            "status": "success",
            "query_result": {
                "original_query": result.query,
                "summary": result.summary,
                "total_matches": result.total_matches,
                "results": result.results
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/explain/issue")
async def explain_code_issue(request: dict):
    """Get natural language explanation for a code issue."""
    try:
        issue = request.get("issue", "")
        context = request.get("context", "")
        
        if not issue:
            raise HTTPException(status_code=400, detail="Issue parameter is required")
        
        explainer = get_issue_explainer()
        explanation = explainer.explain_issue(issue, context)
        
        return {
            "status": "success",
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.get("/templates")
async def list_project_templates():
    """List all available ML/RL project templates."""
    try:
        templates = project_generator.list_templates()
        return {
            "status": "success",
            "templates": templates,
            "total_count": len(templates)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")

@app.get("/templates/{template_name}")
async def get_template_details(template_name: str):
    """Get detailed information about a specific project template."""
    try:
        details = project_generator.get_template_details(template_name)
        return {
            "status": "success",
            "template": details
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get template details: {str(e)}")

@app.post("/templates/generate")
async def generate_project(request: dict):
    """Generate a complete ML project from template."""
    try:
        template_name = request.get("template_name") or request.get("template")
        project_path = request.get("project_path")
        custom_config = request.get("custom_config") or request.get("config", {})
        
        if not template_name:
            raise HTTPException(status_code=400, detail="Template name is required")
        if not project_path:
            raise HTTPException(status_code=400, detail="Project path is required")
        
        result = project_generator.generate_project(
            template_name=template_name,
            project_path=project_path,
            custom_config=custom_config
        )
        
        return {
            "status": "success",
            "project": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate project: {str(e)}")

@app.post("/templates/preview")
async def preview_project_structure(request: dict):
    """Preview what files and directories would be created for a template."""
    try:
        template_name = request.get("template")
        
        if not template_name:
            raise HTTPException(status_code=400, detail="Template name is required")
        
        details = project_generator.get_template_details(template_name)
        
        # Generate requirements preview
        template_obj = project_generator.templates[template_name]
        requirements_content = project_generator._generate_requirements(template_obj)
        
        return {
            "status": "success",
            "preview": {
                "template_name": details["name"],
                "framework": details["framework"],
                "files_to_create": details["files"],
                "directories_to_create": details["directories"],
                "dependencies_count": len(details["dependencies"]),
                "setup_commands": details["setup_commands"],
                "requirements_preview": requirements_content.split('\n')[:10]  # First 10 lines
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preview project: {str(e)}")

@app.get("/privacy-policy")
async def privacy_policy():
    """Serve the privacy policy for the CodeGuard API service."""
    privacy_path = "privacy-policy.md"
    if os.path.exists(privacy_path):
        return FileResponse(privacy_path, media_type="text/markdown")
    else:
        raise HTTPException(status_code=404, detail="Privacy policy not found")

@app.get("/terms-of-service")
async def terms_of_service():
    """Serve the terms of service for the CodeGuard API service."""
    terms_path = "terms-of-service.md"
    if os.path.exists(terms_path):
        return FileResponse(terms_path, media_type="text/markdown")
    else:
        raise HTTPException(status_code=404, detail="Terms of service not found")

@app.post("/improve/code")
async def improve_code_with_ai(request: dict):
    """
    Use AI (OpenAI, Gemini, or Claude) to implement CodeGuard suggestions and improve code quality.
    
    Supports multiple AI providers with fallback functionality.
    """
    try:
        # Extract request data
        original_code = request.get("original_code", request.get("code", ""))
        filename = request.get("filename", "code.py")
        issues = request.get("issues", [])
        fixes = request.get("fixes", [])
        improvement_level = request.get("improvement_level", "moderate")
        ai_provider = request.get("ai_provider", "openai")
        ai_api_key = request.get("ai_api_key")
        
        if not original_code:
            raise HTTPException(status_code=400, detail="Code content is required")
        
        # Convert dict issues/fixes to model objects for compatibility
        from models import Issue, Fix
        issue_objects = [
            Issue(
                filename=issue.get("filename", filename),
                line=issue.get("line", 1),
                type=issue.get("type", "unknown"),
                description=issue.get("description", ""),
                source=issue.get("source", "unknown"),
                severity=issue.get("severity", "warning")
            ) for issue in issues
        ]
        
        fix_objects = [
            Fix(
                filename=fix.get("filename", filename),
                line=fix.get("line", 1),
                suggestion=fix.get("suggestion", ""),
                diff=fix.get("diff"),
                replacement_code=fix.get("replacement_code"),
                auto_fixable=fix.get("auto_fixable", False)
            ) for fix in fixes
        ]
        
        # Use multi-AI manager for better performance and fallback
        multi_ai = get_multi_ai_manager()
        improvement_request = CodeImprovementRequest(
            original_code=original_code,
            filename=filename,
            issues=issue_objects,
            fixes=fix_objects,
            improvement_level=improvement_level,
            preserve_functionality=True
        )
        
        response = await multi_ai.improve_code_with_provider(
            improvement_request, 
            provider_name=ai_provider,
            api_key=ai_api_key
        )
        
        return {
            "improved_code": response.improved_code,
            "applied_fixes": response.applied_fixes,
            "improvement_summary": response.improvement_summary,
            "confidence_score": response.confidence_score,
            "warnings": response.warnings,
            "original_issues_count": len(issues),
            "fixes_applied_count": len(response.applied_fixes),
            "fix_categories": _categorize_fixes_by_type(issues)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code improvement failed: {str(e)}")

def _categorize_fixes_by_type(issues):
    """Categorize issues by type for bulk fixing"""
    categories = {}
    for issue in issues:
        issue_type = issue.get("type", "unknown")
        if issue_type not in categories:
            categories[issue_type] = {
                "count": 0,
                "lines": [],
                "description": issue.get("description", ""),
                "severity": issue.get("severity", "warning")
            }
        categories[issue_type]["count"] += 1
        categories[issue_type]["lines"].append(issue.get("line", 0))
    return categories

@app.post("/improve/bulk-fix")
async def apply_bulk_fixes(request: dict):
    """
    Apply all fixes of a specific type using AI.
    Allows bulk fixing of similar issues across the codebase.
    """
    try:
        original_code = request.get("original_code", "")
        filename = request.get("filename", "code.py")
        fix_type = request.get("fix_type", "")
        issues = request.get("issues", [])
        ai_provider = request.get("ai_provider", "openai")
        ai_api_key = request.get("ai_api_key")
        
        if not original_code or not fix_type:
            raise HTTPException(status_code=400, detail="Code content and fix type are required")
        
        # Filter issues by the specified type
        filtered_issues = [issue for issue in issues if issue.get("type") == fix_type]
        
        if not filtered_issues:
            return {
                "improved_code": original_code,
                "applied_fixes": [],
                "improvement_summary": f"No issues of type '{fix_type}' found",
                "confidence_score": 1.0,
                "warnings": [],
                "fixed_lines": []
            }
        
        # Create focused improvement request for this fix type
        from models import Issue, Fix
        issue_objects = [
            Issue(
                filename=issue.get("filename", filename),
                line=issue.get("line", 1),
                type=issue.get("type", fix_type),
                description=issue.get("description", ""),
                source=issue.get("source", "unknown"),
                severity=issue.get("severity", "warning")
            ) for issue in filtered_issues
        ]
        
        improvement_request = CodeImprovementRequest(
            original_code=original_code,
            filename=filename,
            issues=issue_objects,
            fixes=[],
            improvement_level="moderate",
            preserve_functionality=True
        )
        
        # Use multi-AI manager for bulk fixing
        multi_ai = get_multi_ai_manager()
        response = await multi_ai.improve_code_with_provider(
            improvement_request, 
            provider_name=ai_provider,
            api_key=ai_api_key
        )
        
        return {
            "improved_code": response.improved_code,
            "applied_fixes": response.applied_fixes,
            "improvement_summary": f"Applied bulk fixes for {len(filtered_issues)} instances of '{fix_type}': {response.improvement_summary}",
            "confidence_score": response.confidence_score,
            "warnings": response.warnings,
            "fixed_lines": [issue.get("line") for issue in filtered_issues],
            "fix_type": fix_type,
            "instances_fixed": len(filtered_issues)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk fix failed: {str(e)}")

@app.post("/improve/fim-completion")
async def fim_code_completion(request: dict) -> dict:
    """DeepSeek FIM (Fill In the Middle) completion endpoint for targeted code improvements."""
    try:
        prefix = request.get("prefix", "")
        suffix = request.get("suffix", "")
        ai_provider = request.get("ai_provider", "deepseek")
        ai_api_key = request.get("ai_api_key")
        max_tokens = request.get("max_tokens", 2000)
        
        if not prefix:
            raise HTTPException(status_code=400, detail="Prefix is required for FIM completion")
        
        if ai_provider.lower() != "deepseek":
            raise HTTPException(status_code=400, detail="FIM completion currently only supports DeepSeek")
        
        # Use DeepSeek FIM completion
        improver = get_code_improver()
        
        # Create a prompt that will trigger FIM completion
        fim_prompt = f"```python\n{prefix}\n# TODO: Complete implementation\n{suffix}\n```"
        
        # Create a basic improvement request for FIM
        improvement_request = CodeImprovementRequest(
            original_code=fim_prompt,
            filename="fim_completion.py",
            issues=[],
            fixes=[],
            improvement_level="moderate",
            ai_provider=ai_provider,
            ai_api_key=ai_api_key
        )
        
        # Call the FIM-enhanced DeepSeek integration
        api_key = ai_api_key or os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise HTTPException(status_code=400, detail="DeepSeek API key required for FIM completion")
            
        response = improver._call_deepseek_fim_completion(fim_prompt, api_key)
        
        # Parse response
        import json
        try:
            result = json.loads(response)
            completed_code = result.get("improved_code", "")
            
            # Extract just the completion part
            if "```python" in completed_code:
                start = completed_code.find("```python") + 9
                end = completed_code.find("```", start)
                if end != -1:
                    completed_code = completed_code[start:end].strip()
            
            return {
                "prefix": prefix,
                "suffix": suffix,
                "completion": completed_code,
                "confidence_score": result.get("confidence_score", 0.85),
                "applied_fixes": result.get("applied_fixes", []),
                "warnings": result.get("warnings", [])
            }
            
        except json.JSONDecodeError:
            # If response isn't JSON, return raw completion
            return {
                "prefix": prefix,
                "suffix": suffix,
                "completion": response,
                "confidence_score": 0.8,
                "applied_fixes": [],
                "warnings": ["Raw completion returned"]
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FIM completion failed: {str(e)}")

@app.post("/reports/improvement-analysis")
async def generate_improvement_report(request: dict):
    """
    Generate a comprehensive improvement report showing all issues with original code,
    line numbers, and detailed recommendations for fixes.
    """
    try:
        files = request.get("files", [])
        include_ai_suggestions = request.get("include_ai_suggestions", True)
        report_format = request.get("format", "markdown")  # markdown, json, html
        apply_filtering = request.get("apply_false_positive_filtering", True)  # Filter out common false positives
        
        if not files:
            raise HTTPException(status_code=400, detail="Files are required for analysis")
        
        # Perform comprehensive audit using existing audit endpoint
        audit_data = {
            "files": files,
            "options": {
                "analysis_level": "strict",
                "framework": "general",
                "target_platform": "general"
            }
        }
        
        # Use the existing audit engine with false positive filtering
        from enhanced_audit import EnhancedAuditEngine
        from models import AuditRequest, CodeFile, AuditOptions
        from false_positive_filter import get_false_positive_filter
        
        audit_engine = EnhancedAuditEngine(use_false_positive_filter=True)
        audit_request = AuditRequest(
            files=[CodeFile(filename=f["filename"], content=f["content"]) for f in files],
            options=AuditOptions(level="strict", framework="general", target="general")
        )
        
        # Get raw audit results
        audit_response = audit_engine.analyze_code(audit_request)
        
        # Apply false positive filtering if requested (uses fast rule-based filtering, no ChatGPT calls)
        if apply_filtering:
            false_positive_filter = get_false_positive_filter()
            filtered_issues, filtered_fixes = false_positive_filter.filter_issues(
                audit_response.issues, 
                audit_response.fixes, 
                audit_request.files
            )
            
            # Update audit response with filtered results
            audit_response.issues = filtered_issues
            audit_response.fixes = filtered_fixes
            audit_response.summary = f"Found {len(filtered_issues)} issues across {len(files)} files (after filtering)"
        else:
            audit_response.summary = f"Found {len(audit_response.issues)} issues across {len(files)} files (unfiltered)"
        
        # Generate AI improvement suggestions if requested
        ai_suggestions = {}
        if include_ai_suggestions:
            multi_ai = get_multi_ai_manager()
            for file_data in files:
                filename = file_data["filename"]
                content = file_data["content"]
                
                # Get file-specific issues
                file_issues = [issue for issue in audit_response.issues if issue.filename == filename]
                
                if file_issues:
                    # Create improvement request
                    improvement_request = CodeImprovementRequest(
                        original_code=content,
                        filename=filename,
                        issues=file_issues[:10],  # Limit to top 10 issues for performance
                        fixes=[],
                        improvement_level="moderate"
                    )
                    
                    try:
                        ai_response = await multi_ai.improve_code_with_provider(improvement_request, "openai")
                        ai_suggestions[filename] = {
                            "improved_code": ai_response.improved_code,
                            "summary": ai_response.improvement_summary,
                            "confidence": ai_response.confidence_score
                        }
                    except Exception as e:
                        ai_suggestions[filename] = {
                            "error": f"AI analysis failed: {str(e)}",
                            "improved_code": content,
                            "summary": "Manual review recommended",
                            "confidence": 0.0
                        }
        
        # Generate the report
        report_data = _generate_improvement_report_data(audit_response, files, ai_suggestions)
        
        if report_format == "markdown":
            formatted_report = _format_report_as_markdown(report_data)
        elif report_format == "html":
            formatted_report = _format_report_as_html(report_data)
        else:  # json
            formatted_report = report_data
        
        return {
            "report": formatted_report,
            "format": report_format,
            "total_issues": len(audit_response.issues),
            "total_files": len(files),
            "severity_breakdown": _get_severity_breakdown(audit_response.issues),
            "issue_categories": _categorize_fixes_by_type([issue.__dict__ for issue in audit_response.issues]),
            "ai_suggestions_included": include_ai_suggestions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

def _generate_improvement_report_data(audit_response, files, ai_suggestions):
    """Generate structured report data"""
    files_dict = {f["filename"]: f["content"] for f in files}
    
    report_data = {
        "title": "CodeGuard Improvement Analysis Report",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_files": len(files),
            "total_issues": len(audit_response.issues),
            "total_fixes": len(audit_response.fixes)
        },
        "files": []
    }
    
    # Group issues by file
    files_with_issues = {}
    for issue in audit_response.issues:
        filename = issue.filename
        if filename not in files_with_issues:
            files_with_issues[filename] = []
        files_with_issues[filename].append(issue)
    
    # Process each file
    for filename, content in files_dict.items():
        file_issues = files_with_issues.get(filename, [])
        lines = content.split('\n')
        
        file_data = {
            "filename": filename,
            "line_count": len(lines),
            "issue_count": len(file_issues),
            "issues": [],
            "ai_suggestion": ai_suggestions.get(filename, {})
        }
        
        # Process each issue
        for issue in file_issues:
            line_num = issue.line
            original_line = lines[line_num - 1] if 0 < line_num <= len(lines) else ""
            
            # Get context (2 lines before and after)
            context_start = max(0, line_num - 3)
            context_end = min(len(lines), line_num + 2)
            context_lines = []
            
            for i in range(context_start, context_end):
                prefix = ">>> " if i == line_num - 1 else "    "
                context_lines.append(f"{prefix}{i+1:3d}: {lines[i]}")
            
            issue_data = {
                "line": line_num,
                "type": issue.type,
                "severity": issue.severity,
                "source": issue.source,
                "description": issue.description,
                "original_code": original_line.strip(),
                "context": "\n".join(context_lines),
                "recommendations": _get_issue_recommendations(issue)
            }
            
            file_data["issues"].append(issue_data)
        
        # Sort issues by line number
        file_data["issues"].sort(key=lambda x: x["line"])
        report_data["files"].append(file_data)
    
    return report_data

def _get_issue_recommendations(issue):
    """Get specific recommendations for different issue types"""
    recommendations = {
        "syntax": "Fix syntax error according to Python language specification",
        "style": "Follow PEP 8 style guidelines for consistent code formatting",
        "security": "Address security vulnerability to prevent potential exploits",
        "performance": "Optimize code for better runtime performance",
        "ml_pattern": "Apply machine learning best practices and patterns",
        "rl_pattern": "Follow reinforcement learning coding standards"
    }
    
    # Get specific recommendations based on issue details
    if "import" in issue.description.lower():
        return "Organize imports according to PEP 8: standard library, third-party, local imports"
    elif "unused" in issue.description.lower():
        return "Remove unused variables/imports to improve code cleanliness"
    elif "line too long" in issue.description.lower():
        return "Break long lines at 88 characters using parentheses or line continuation"
    elif "missing docstring" in issue.description.lower():
        return "Add descriptive docstrings following Google or NumPy style"
    elif "complexity" in issue.description.lower():
        return "Reduce cyclomatic complexity by extracting functions or simplifying logic"
    
    return recommendations.get(issue.type, "Review and fix according to coding standards")

def _format_report_as_markdown(report_data):
    """Format report data as Markdown"""
    md = f"""# {report_data['title']}

**Generated:** {report_data['generated_at']}

## Summary
- **Files Analyzed:** {report_data['summary']['total_files']}
- **Total Issues:** {report_data['summary']['total_issues']}
- **Total Fixes:** {report_data['summary']['total_fixes']}

"""
    
    for file_data in report_data['files']:
        if file_data['issue_count'] == 0:
            continue
            
        md += f"""## File: `{file_data['filename']}`
**Lines:** {file_data['line_count']} | **Issues:** {file_data['issue_count']}

"""
        
        # Add AI suggestion if available
        if file_data.get('ai_suggestion', {}).get('summary'):
            ai_data = file_data['ai_suggestion']
            md += f"""### AI Analysis Summary
**Confidence:** {ai_data.get('confidence', 0):.1%}
**Recommendations:** {ai_data.get('summary', 'No summary available')}

"""
        
        # Add issues
        for i, issue in enumerate(file_data['issues'], 1):
            severity_icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(issue['severity'], "⚪")
            
            md += f"""### Issue #{i}: Line {issue['line']} {severity_icon}
**Type:** {issue['type']} | **Source:** {issue['source']} | **Severity:** {issue['severity']}

**Problem:** {issue['description']}

**Original Code:**
```python
{issue['original_code']}
```

**Context:**
```python
{issue['context']}
```

**Recommendation:** {issue['recommendations']}

---

"""
    
    return md

def _format_report_as_html(report_data):
    """Format report data as HTML"""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report_data['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
        .file-section {{ margin: 30px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .file-header {{ background: #e9ecef; padding: 15px; font-weight: bold; }}
        .issue {{ margin: 15px; padding: 15px; border-left: 4px solid #007bff; }}
        .issue.error {{ border-left-color: #dc3545; }}
        .issue.warning {{ border-left-color: #ffc107; }}
        .code {{ background: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; }}
        .severity {{ padding: 3px 8px; border-radius: 3px; color: white; font-size: 12px; }}
        .severity.error {{ background: #dc3545; }}
        .severity.warning {{ background: #ffc107; color: black; }}
        .severity.info {{ background: #17a2b8; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_data['title']}</h1>
        <p><strong>Generated:</strong> {report_data['generated_at']}</p>
        <p><strong>Files:</strong> {report_data['summary']['total_files']} | 
           <strong>Issues:</strong> {report_data['summary']['total_issues']}</p>
    </div>
"""
    
    for file_data in report_data['files']:
        if file_data['issue_count'] == 0:
            continue
            
        html += f"""
    <div class="file-section">
        <div class="file-header">
            📄 {file_data['filename']} 
            <span style="float: right;">{file_data['issue_count']} issues</span>
        </div>
"""
        
        for issue in file_data['issues']:
            html += f"""
        <div class="issue {issue['severity']}">
            <h4>Line {issue['line']}: {issue['type']} 
                <span class="severity {issue['severity']}">{issue['severity'].upper()}</span>
            </h4>
            <p><strong>Problem:</strong> {issue['description']}</p>
            <p><strong>Original Code:</strong></p>
            <div class="code">{issue['original_code']}</div>
            <p><strong>Recommendation:</strong> {issue['recommendations']}</p>
        </div>
"""
        
        html += "    </div>"
    
    html += """
</body>
</html>"""
    
    return html

def _get_severity_breakdown(issues):
    """Get breakdown of issues by severity"""
    breakdown = {"error": 0, "warning": 0, "info": 0}
    for issue in issues:
        severity = issue.severity.lower()
        if severity in breakdown:
            breakdown[severity] += 1
    return breakdown

@app.post("/improve/project")
async def improve_entire_project(request: dict):
    """
    Improve multiple files in a project using ChatGPT and CodeGuard analysis.
    
    Performs audit on all files first, then applies AI-powered improvements.
    """
    try:
        files = request.get("files", [])
        improvement_level = request.get("improvement_level", "moderate")
        
        if not files:
            raise HTTPException(status_code=400, detail="At least one file is required")
        
        # Convert to CodeFile objects
        from models import CodeFile, AuditRequest
        code_files = [
            CodeFile(filename=f.get("filename", ""), content=f.get("content", ""))
            for f in files
        ]
        
        # First, run complete audit on all files
        audit_request = AuditRequest(files=code_files)
        audit_response = analyze_code(audit_request)
        
        # Then improve each file using ChatGPT
        batch_improver = get_batch_improver()
        improvements = batch_improver.improve_project(code_files, {
            "issues": audit_response.issues,
            "fixes": audit_response.fixes
        })
        
        # Compile results
        project_results = {
            "original_audit": {
                "summary": audit_response.summary,
                "total_issues": len(audit_response.issues),
                "total_fixes": len(audit_response.fixes)
            },
            "improvements": {},
            "overall_summary": {
                "files_processed": len(files),
                "files_improved": 0,
                "total_fixes_applied": 0,
                "average_confidence": 0.0
            }
        }
        
        total_confidence = 0.0
        files_improved = 0
        total_fixes_applied = 0
        
        for filename, improvement in improvements.items():
            project_results["improvements"][filename] = {
                "improved_code": improvement.improved_code,
                "applied_fixes": improvement.applied_fixes,
                "improvement_summary": improvement.improvement_summary,
                "confidence_score": improvement.confidence_score,
                "warnings": improvement.warnings
            }
            
            if improvement.applied_fixes:
                files_improved += 1
                total_fixes_applied += len(improvement.applied_fixes)
            
            total_confidence += improvement.confidence_score
        
        # Update overall summary
        project_results["overall_summary"]["files_improved"] = files_improved
        project_results["overall_summary"]["total_fixes_applied"] = total_fixes_applied
        project_results["overall_summary"]["average_confidence"] = total_confidence / len(files) if files else 0.0
        
        return project_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Project improvement failed: {str(e)}")

@app.post("/audit-and-improve")
async def audit_and_improve_combined(request: AuditRequest):
    """
    Combined endpoint that performs CodeGuard audit and ChatGPT improvements in one call.
    
    This is the most comprehensive endpoint for getting both analysis and AI-powered fixes.
    """
    try:
        # Perform initial audit WITHOUT false positive filtering to get all issues
        audit_response = await _perform_audit(request, validate_with_ai=False)
        
        # Track telemetry for the audit
        session_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Improve each file using AI
        improvements = {}
        batch_improver = get_batch_improver()
        
        if request.files:
            try:
                # Create improvement requests for each file
                for file in request.files:
                    from chatgpt_integration import CodeImprovementRequest
                    
                    improvement_request = CodeImprovementRequest(
                        original_code=file.content,
                        filename=file.filename,
                        issues=audit_response.issues,
                        fixes=audit_response.fixes,
                        improvement_level="moderate",
                        preserve_functionality=True,
                        ai_provider=request.ai_provider or "openai",
                        ai_api_key=request.ai_api_key
                    )
                    
                    code_improver = get_code_improver()
                    improvement = code_improver.improve_code(improvement_request)
                    
                    improvements[file.filename] = {
                        "improved_code": improvement.improved_code,
                        "applied_fixes": improvement.applied_fixes,
                        "improvement_summary": improvement.improvement_summary,
                        "confidence_score": improvement.confidence_score,
                        "warnings": improvement.warnings
                    }
            except Exception as e:
                # If AI improvement fails, apply fallback improvements
                for file in request.files:
                    from chatgpt_integration import CodeImprovementRequest
                    
                    # Create fallback improvement request
                    fallback_request = CodeImprovementRequest(
                        original_code=file.content,
                        filename=file.filename,
                        issues=audit_response.issues,
                        fixes=audit_response.fixes,
                        improvement_level="moderate",
                        preserve_functionality=True,
                        ai_provider="fallback",
                        ai_api_key=None
                    )
                    
                    code_improver = get_code_improver()
                    fallback_improvement = code_improver._fallback_improvement(fallback_request)
                    
                    improvements[file.filename] = {
                        "improved_code": fallback_improvement.improved_code,
                        "applied_fixes": fallback_improvement.applied_fixes,
                        "improvement_summary": f"AI improvement failed: {str(e)}. Applied {len(fallback_improvement.applied_fixes)} automatic fixes.",
                        "confidence_score": fallback_improvement.confidence_score,
                        "warnings": fallback_improvement.warnings + [f"Could not connect to {request.ai_provider or 'AI provider'}: {str(e)}"]
                    }
        
        # Calculate improvement statistics
        total_ai_fixes = sum(len(imp.get("applied_fixes", [])) for imp in improvements.values())
        avg_confidence = sum(imp.get("confidence_score", 0) for imp in improvements.values()) / len(improvements) if improvements else 0.0
        
        # Record telemetry
        processing_time = time.time() - start_time
        framework = "unknown"
        if request.options and request.options.framework:
            framework = request.options.framework
        else:
            # Detect framework from code content
            all_content = " ".join([f.content for f in request.files])
            if "torch" in all_content or "pytorch" in all_content:
                framework = "pytorch"
            elif "tensorflow" in all_content or "tf." in all_content:
                framework = "tensorflow"
            elif "gym" in all_content:
                framework = "gym"
        
        # Record telemetry (skip if errors occur to avoid blocking)
        try:
            from datetime import datetime
            from telemetry import AuditSession
            
            # Get error types and severity breakdown
            error_type_counts = {}
            severity_counts = {}
            for issue in audit_response.issues:
                issue_type = issue.type
                severity = issue.severity
                error_type_counts[issue_type] = error_type_counts.get(issue_type, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            tools_used = list(set(issue.source for issue in audit_response.issues))
            
            session_data = AuditSession(
                session_id=session_id,
                timestamp=datetime.now(),
                file_count=len(request.files),
                total_issues=len(audit_response.issues),
                error_types=error_type_counts,
                severity_breakdown=severity_counts,
                analysis_time_ms=int(processing_time * 1000),
                framework_detected=framework,
                tools_used=tools_used
            )
            telemetry_collector.record_audit_session(session_data)
        except Exception:
            pass  # Don't let telemetry errors block the response
        
        return {
            "audit_results": {
                "summary": audit_response.summary,
                "issues": [issue.dict() for issue in audit_response.issues],
                "fixes": [fix.dict() for fix in audit_response.fixes]
            },
            "ai_improvements": improvements,
            "combined_summary": {
                "session_id": session_id,
                "total_issues_found": len(audit_response.issues),
                "codeguard_fixes": len(audit_response.fixes),
                "ai_fixes_applied": total_ai_fixes,
                "average_ai_confidence": avg_confidence,
                "processing_time_seconds": round(processing_time, 2),
                "framework_detected": framework
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Combined audit and improvement failed: {str(e)}")

if __name__ == "__main__":
    # Run the application - optimized for both development and Cloud Run
    environment = os.environ.get("ENVIRONMENT", "development")
    
    # Set appropriate default port based on environment
    if environment == "production":
        default_port = 8080  # Cloud Run standard port
    else:
        default_port = 5000  # Replit development port
    
    port = int(os.environ.get("PORT", default_port))
    
    print(f"Starting CodeGuard API on 0.0.0.0:{port} (environment: {environment})")
    
    if environment == "development":
        # Use import string for reload functionality in development
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=True,
            log_level="info",
            access_log=True
        )
    else:
        # Use app object directly in production for better performance
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="info",
            access_log=True
        )
