# CodeGuard API - Complete Usage Guide

## Overview

CodeGuard API is a FastAPI-based service that analyzes Python code for ML/RL projects, providing structured reports of issues and fix suggestions. The API includes secure authentication and is fully compatible with OpenAI GPT Actions.

## HTTPS Endpoint

**Production URL**: `https://87ee31f3-2ea8-47fa-bfc6-dab95a535424-00-2j7aj3sdppcjx.riker.replit.dev`

## API Key Authentication

The CodeGuard API includes secure API key authentication to protect the code analysis service.

### How Authentication Works

1. **Development Mode**: If no `CODEGUARD_API_KEY` environment variable is set, the API allows open access for testing
2. **Production Mode**: When `CODEGUARD_API_KEY` is configured, all requests to `/audit` require a valid Bearer token

### Setting Up Authentication

To enable API key protection, set the environment variable:
```bash
export CODEGUARD_API_KEY="your-secure-api-key-here"
```

### Using the API with Authentication

#### Making Authenticated Requests

Include your API key in the Authorization header:

```bash
curl -X POST "https://87ee31f3-2ea8-47fa-bfc6-dab95a535424-00-2j7aj3sdppcjx.riker.replit.dev/audit" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-api-key" \
     -d '{
       "files": [
         {
           "filename": "example.py",
           "content": "import torch\nprint(hello)"
         }
       ]
     }'
```

#### Authentication Status Check

Verify your API key is working:

```bash
curl -X GET "https://87ee31f3-2ea8-47fa-bfc6-dab95a535424-00-2j7aj3sdppcjx.riker.replit.dev/auth/status" \
     -H "Authorization: Bearer your-api-key"
```

### API Endpoints

#### Protected Endpoints (Require API Key)
- `POST /audit` - Code analysis
- `GET /auth/status` - Authentication verification

#### Public Endpoints (No Authentication)
- `GET /` - Service information
- `GET /health` - Health check
- `GET /.well-known/openapi.yaml` - API specification
- `GET /docs` - Interactive documentation

### Quick Test URLs

- **API Homepage**: `https://87ee31f3-2ea8-47fa-bfc6-dab95a535424-00-2j7aj3sdppcjx.riker.replit.dev/`
- **Interactive Docs**: `https://87ee31f3-2ea8-47fa-bfc6-dab95a535424-00-2j7aj3sdppcjx.riker.replit.dev/docs`
- **Health Check**: `https://87ee31f3-2ea8-47fa-bfc6-dab95a535424-00-2j7aj3sdppcjx.riker.replit.dev/health`
- **OpenAPI Spec**: `https://87ee31f3-2ea8-47fa-bfc6-dab95a535424-00-2j7aj3sdppcjx.riker.replit.dev/.well-known/openapi.yaml`

### Error Responses

#### 401 Unauthorized
```json
{
  "detail": "Invalid API key"
}
```

#### 403 Forbidden
```json
{
  "detail": "Not authenticated"
}
```

### Security Features

- **Secure Comparison**: Uses `hmac.compare_digest()` to prevent timing attacks
- **API Key Hashing**: Logs only partial hashes for security
- **Bearer Token Standard**: Follows HTTP Bearer authentication standard
- **Environment Variable Config**: API keys stored securely in environment variables

### OpenAI GPT Action Integration

The API includes Bearer authentication in the OpenAPI specification, making it compatible with OpenAI Actions that require authenticated endpoints.

Example GPT Action configuration:
```yaml
authentication:
  type: bearer
  token: "your-api-key"
```