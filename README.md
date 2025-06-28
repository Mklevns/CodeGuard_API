# CodeGuard API - Authentication Guide

## API Key Authentication

The CodeGuard API now includes secure API key authentication to protect the code analysis service.

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
curl -X POST "http://34.55.167.13:5000/audit" \
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
curl -X GET "http://34.55.167.13:5000/auth/status" \
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