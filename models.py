from pydantic import BaseModel, Field
from typing import List, Optional

class CodeFile(BaseModel):
    """Represents a single code file to be analyzed."""
    filename: str = Field(description="Name of the file", examples=["train.py"])
    content: str = Field(description="Content of the file", examples=["import torch\n\n# TODO: training code"])

class AuditOptions(BaseModel):
    """Optional configuration for the audit process."""
    level: Optional[str] = Field(default="strict", description="Analysis level", examples=["strict"])
    framework: Optional[str] = Field(default="pytorch", description="ML framework", examples=["pytorch"])
    target: Optional[str] = Field(default="gpu", description="Target platform", examples=["gpu"])

class AuditRequest(BaseModel):
    """Request model for the audit endpoint."""
    files: List[CodeFile] = Field(description="List of files to analyze", min_length=1)
    options: Optional[AuditOptions] = Field(default=None, description="Optional audit configuration")

    class Config:
        json_schema_extra = {
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
    filename: str = Field(description="File where the issue was found")
    line: int = Field(description="Line number of the issue")
    type: str = Field(description="Type of issue", examples=["style"])
    description: str = Field(description="Description of the issue")
    source: str = Field(description="Analysis tool that detected the issue", examples=["flake8", "pylint", "mypy", "ml_rules"])
    severity: str = Field(description="Issue severity level", examples=["error", "warning", "info"], default="warning")

class Fix(BaseModel):
    """Represents a suggested fix for an issue."""
    filename: str = Field(description="File where the fix should be applied")
    line: int = Field(description="Line number for the fix")
    suggestion: str = Field(description="Suggested fix")
    diff: Optional[str] = Field(description="Unified diff showing the change", default=None)
    replacement_code: Optional[str] = Field(description="Complete replacement code for the line/block", default=None)
    auto_fixable: bool = Field(description="Whether this fix can be applied automatically", default=False)

class AuditResponse(BaseModel):
    """Response model for the audit endpoint."""
    summary: str = Field(description="Summary of the audit results")
    issues: List[Issue] = Field(description="List of issues found")
    fixes: List[Fix] = Field(description="List of suggested fixes")

    class Config:
        json_schema_extra = {
            "example": {
                "summary": "5 issues found across 2 files (1 errors, 4 warnings)",
                "issues": [
                    {
                        "filename": "train.py",
                        "line": 1,
                        "type": "style",
                        "description": "F401: Unused import 'torch'",
                        "source": "flake8",
                        "severity": "warning"
                    },
                    {
                        "filename": "train.py",
                        "line": 5,
                        "type": "best_practice",
                        "description": "Missing random seeding for reproducibility",
                        "source": "ml_rules",
                        "severity": "warning"
                    },
                    {
                        "filename": "utils.py",
                        "line": 1,
                        "type": "style",
                        "description": "Import statements can be better organized",
                        "source": "isort",
                        "severity": "info"
                    }
                ],
                "fixes": [
                    {
                        "filename": "train.py",
                        "line": 1,
                        "suggestion": "Remove unused import 'torch'",
                        "diff": "- import torch\n+",
                        "auto_fixable": True
                    },
                    {
                        "filename": "train.py",
                        "line": 1,
                        "suggestion": "Add seeding for reproducibility",
                        "replacement_code": "torch.manual_seed(42)\nrandom.seed(42)",
                        "auto_fixable": True
                    }
                ]
            }
        }
