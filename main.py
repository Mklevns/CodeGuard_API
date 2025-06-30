from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from datetime import datetime
from models import AuditRequest, AuditResponse
from audit import analyze_code
from auth import verify_api_key, get_current_user
from analysis_cache import get_file_cache, get_project_cache
from granular_rule_config import get_rule_manager
from rule_loader import CustomRuleEngine
from telemetry import telemetry_collector, metrics_analyzer
from dashboard import get_dashboard
from historical_timeline import get_timeline_generator
from gpt_connector import get_gpt_connector, get_issue_explainer
from project_templates import MLProjectGenerator
from chatgpt_integration import get_code_improver, get_batch_improver, CodeImprovementRequest
from github_repo_context import get_repo_context_provider
from llm_prompt_generator import get_llm_prompt_generator
from multi_ai_integration import get_multi_ai_manager
from ml_performance_heatmap import heatmap_api, HeatmapConfig
from git_analyzer import analyze_git_history, GitContextRetriever
from graph_analyzer import analyze_repository_structure
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

# Initialize project generator and Git context retriever
project_generator = MLProjectGenerator()
git_context_retriever = GitContextRetriever()

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

async def _run_shared_audit(request: AuditRequest, validate_with_ai: bool = True) -> AuditResponse:
    """
    Shared audit logic used by multiple endpoints.

    Args:
        request: AuditRequest containing files to analyze
        validate_with_ai: Whether to apply ChatGPT false positive filtering

    Returns:
        AuditResponse with analysis results
    """
    return await _perform_audit(request, validate_with_ai=validate_with_ai)

@app.post("/audit", response_model=AuditResponse)
async def audit_code(request: AuditRequest, current_user: dict = Depends(get_current_user)):
    """
    Standard audit endpoint - includes ChatGPT false positive filtering by default.
    """
    return await _run_shared_audit(request, validate_with_ai=True)

@app.post("/audit/no-filter", response_model=AuditResponse)
async def audit_code_no_filter(request: AuditRequest, current_user: dict = Depends(get_current_user)):
    """
    Audit endpoint without ChatGPT false positive filtering for debugging.
    """
    return await _run_shared_audit(request, validate_with_ai=False)

def _record_audit_telemetry(session_id: str, request: AuditRequest, response: AuditResponse, analysis_time: float):
    """Record telemetry data for audit session."""
    try:
        request_analysis = metrics_analyzer.analyze_request(request)
        response_analysis = metrics_analyzer.analyze_response(response)

        session = metrics_analyzer.create_session(
            session_id, request_analysis, response_analysis, analysis_time
        )
        telemetry_collector.record_audit_session(session)
        telemetry_collector.record_error_patterns(session_id, response.issues)

        if request_analysis['primary_framework']:
            telemetry_collector.record_framework_usage(request_analysis['primary_framework'])
    except Exception:
        # Don't let telemetry errors affect audit results
        pass

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
        # Perform code analysis with configurable AI validation and timeout
        from enhanced_audit import EnhancedAuditEngine
        import asyncio

        engine = EnhancedAuditEngine(use_false_positive_filter=validate_with_ai)
        analysis_timeout = 40  # Total analysis timeout in seconds

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

        # Record telemetry
        analysis_time = (time.time() - start_time) * 1000
        _record_audit_telemetry(session_id, request, response, analysis_time)

        return response

    except Exception as e:
        analysis_time = (time.time() - start_time) * 1000
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

@app.post("/improve")
async def improve_code_universal(request: dict):
    """
    UNIVERSAL CODE IMPROVEMENT: RESTful endpoint that accepts optional audit results.

    If audit_results are provided:
    - Applies targeted improvements to known issues (fast)
    - Preserves code structure by default

    If audit_results are NOT provided:
    - Runs full audit first, then applies improvements (comprehensive)
    - Equivalent to the old /audit-and-improve endpoint
    """
    try:
        # Extract request data
        original_code = request.get("original_code", request.get("code", ""))
        filename = request.get("filename", "code.py")
        audit_results = request.get("audit_results")  # Optional: pre-existing audit results
        improvement_level = request.get("improvement_level", "moderate")
        ai_provider = request.get("ai_provider", "openai")
        ai_api_key = request.get("ai_api_key")
        target_lines = request.get("target_lines", [])
        preserve_structure = request.get("preserve_structure", True)

        if not original_code:
            raise HTTPException(status_code=400, detail="Code content is required")

        # If audit results not provided, run audit first
        if not audit_results:
            from models import CodeFile, AuditOptions

            # Create audit request
            audit_request = AuditRequest(
                files=[CodeFile(filename=filename, content=original_code)],
                options=AuditOptions(
                    level=request.get("analysis_level", "strict"),
                    framework=request.get("framework", "auto"),
                    target=request.get("target", "gpu")
                ),
                ai_provider=ai_provider,
                ai_api_key=ai_api_key
            )

            # Run shared audit logic
            audit_response = await _run_shared_audit(audit_request, validate_with_ai=True)

            # Extract issues and fixes
            issues = [issue.dict() for issue in audit_response.issues]
            fixes = [fix.dict() for fix in audit_response.fixes]

            improvement_mode = "comprehensive"
        else:
            # Use provided audit results
            issues = audit_results.get("issues", [])
            fixes = audit_results.get("fixes", [])
            improvement_mode = "targeted"

            if not issues:
                return {
                    "improved_code": original_code,
                    "applied_fixes": [],
                    "improvement_summary": "No issues found in provided audit results",
                    "confidence_score": 1.0,
                    "warnings": [],
                    "improvement_mode": improvement_mode
                }

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

        # Filter issues to only requested lines if specified
        if target_lines:
            issue_objects = [issue for issue in issue_objects if issue.line in target_lines]

        # Use multi-AI manager for better performance and fallback
        multi_ai = get_multi_ai_manager()
        improvement_request = CodeImprovementRequest(
            original_code=original_code,
            filename=filename,
            issues=issue_objects,
            fixes=fix_objects,
            improvement_level="conservative" if preserve_structure else improvement_level,
            preserve_functionality=preserve_structure
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
            "fix_categories": _categorize_fixes_by_type(issues),
            "improvement_mode": improvement_mode,
            "audit_was_run": audit_results is None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code improvement failed: {str(e)}")

@app.post("/improve/code")
async def improve_code_with_ai(request: dict):
    """
    LEGACY ENDPOINT: Redirects to /improve with targeted mode.
    Maintained for backward compatibility.
    """
    # Add audit_results to force targeted mode
    if "audit_results" not in request and "issues" in request:
        request["audit_results"] = {
            "issues": request.get("issues", []),
            "fixes": request.get("fixes", [])
        }

    return await improve_code_universal(request)

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

@app.post("/improve/generate-custom-prompt")
async def generate_custom_prompt(request: dict) -> dict:
    """Generate a custom AI prompt based on audit results using LLM analysis."""
    try:
        files = request.get("files", [])
        issues = request.get("issues", [])
        fixes = request.get("fixes", [])
        ai_provider = request.get("ai_provider", "openai")

        if not files:
            raise HTTPException(status_code=400, detail="Files are required for prompt generation")

        # Convert to proper objects
        from models import CodeFile, Issue, Fix

        code_files = [CodeFile(filename=f.get("filename", ""), content=f.get("content", "")) for f in files]
        issue_objects = [Issue(
            filename=i.get("filename", ""),
            line=i.get("line", 1),
            type=i.get("type", "unknown"),
            description=i.get("description", ""),
            source=i.get("source", "unknown"),
            severity=i.get("severity", "warning")
        ) for i in issues]

        # Generate custom prompt using LLM
        prompt_generator = get_llm_prompt_generator()
        result = prompt_generator.generate_custom_prompt(
            issues=issue_objects,
            fixes=[],
            code_files=code_files,
            ai_provider=ai_provider
        )

        return {
            "status": "success",
            "custom_prompt": result.system_prompt,
            "confidence_boost": result.confidence_boost,
            "focus_areas": result.focus_areas,
            "prompt_strategy": result.prompt_strategy,
            "estimated_effectiveness": result.estimated_effectiveness,
            "total_issues_analyzed": len(issue_objects),
            "frameworks_detected": len([f for f in code_files if any(pattern in f.content for pattern in ["torch", "tensorflow", "gym", "sklearn"])]),
            "ai_provider": ai_provider
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt generation failed: {str(e)}")

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
    Generate a{issue['original_code']}
```

**Context:**
```python
{issue['context']}