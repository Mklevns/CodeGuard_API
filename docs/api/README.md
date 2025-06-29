# CodeGuard API Reference

Complete API documentation for the CodeGuard static analysis platform.

## Base URL

**Production**: `https://codeguard.replit.app`

## Authentication

All protected endpoints require Bearer token authentication:

```http
Authorization: Bearer your-api-key-here
```

## Core Endpoints

### Code Analysis

#### POST `/audit`
Analyze Python code files for ML/RL issues.

**Request:**
```json
{
  "files": [
    {
      "filename": "example.py",
      "content": "import torch\nprint('Hello ML')"
    }
  ],
  "options": {
    "analysis_level": "comprehensive",
    "framework": "pytorch"
  }
}
```

**Response:**
```json
{
  "issues": [...],
  "fixes": [...],
  "summary": {...},
  "telemetry": {...}
}
```

### AI-Powered Improvements

#### POST `/improve/code`
Get AI-powered code improvements.

#### POST `/improve/project`
Bulk improvement for multiple files.

#### POST `/audit-and-improve`
Combined audit and improvement in one call.

### Project Templates

#### GET `/templates`
List available ML/RL project templates.

#### POST `/templates/generate`
Generate new project from template.

### Analytics

#### GET `/metrics/usage`
Usage statistics and performance metrics.

#### GET `/dashboard`
Comprehensive analytics dashboard.

## Error Handling

Standard HTTP status codes with detailed error messages:

- `400` - Bad Request: Invalid input data
- `401` - Unauthorized: Missing or invalid API key
- `422` - Validation Error: Request validation failed
- `500` - Internal Error: Server processing error

## Rate Limits

- 100 requests per minute per API key
- 1000 requests per hour per API key
- File size limit: 1MB per file
- Maximum 10 files per request