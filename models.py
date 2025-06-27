from pydantic import BaseModel, Field
from typing import List, Optional

class CodeFile(BaseModel):
    """Represents a single code file to be analyzed."""
    filename: str = Field(..., description="Name of the file", example="train.py")
    content: str = Field(..., description="Content of the file", example="import torch\n\n# TODO: training code")

class AuditOptions(BaseModel):
    """Optional configuration for the audit process."""
    level: Optional[str] = Field(default="strict", description="Analysis level", example="strict")
    framework: Optional[str] = Field(default="pytorch", description="ML framework", example="pytorch")
    target: Optional[str] = Field(default="gpu", description="Target platform", example="gpu")

class AuditRequest(BaseModel):
    """Request model for the audit endpoint."""
    files: List[CodeFile] = Field(..., description="List of files to analyze", min_items=1)
    options: Optional[AuditOptions] = Field(default=None, description="Optional audit configuration")

    class Config:
        schema_extra = {
            "example": {
                "files": [
                    {
                        "filename": "train.py",
                        "content": "import torch\n\n# TODO: training code"
                    },
                    {
                        "filename": "utils.py",
                        "content": "def seed_everything(seed): pass"
                    }
                ],
                "options": {
                    "level": "strict",
                    "framework": "pytorch",
                    "target": "gpu"
                }
            }
        }

class Issue(BaseModel):
    """Represents a code issue found during analysis."""
    filename: str = Field(..., description="File where the issue was found")
    line: int = Field(..., description="Line number of the issue")
    type: str = Field(..., description="Type of issue", example="style")
    description: str = Field(..., description="Description of the issue")

class Fix(BaseModel):
    """Represents a suggested fix for an issue."""
    filename: str = Field(..., description="File where the fix should be applied")
    line: int = Field(..., description="Line number for the fix")
    suggestion: str = Field(..., description="Suggested fix")

class AuditResponse(BaseModel):
    """Response model for the audit endpoint."""
    summary: str = Field(..., description="Summary of the audit results")
    issues: List[Issue] = Field(..., description="List of issues found")
    fixes: List[Fix] = Field(..., description="List of suggested fixes")

    class Config:
        schema_extra = {
            "example": {
                "summary": "2 issues found across 2 files",
                "issues": [
                    {
                        "filename": "train.py",
                        "line": 1,
                        "type": "style",
                        "description": "Unused import 'torch'"
                    },
                    {
                        "filename": "utils.py",
                        "line": 1,
                        "type": "best_practice",
                        "description": "Missing function docstring"
                    }
                ],
                "fixes": [
                    {
                        "filename": "train.py",
                        "line": 1,
                        "suggestion": "Remove unused import 'torch'"
                    }
                ]
            }
        }
