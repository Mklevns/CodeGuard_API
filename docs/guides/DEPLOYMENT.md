# CodeGuard API - Cloud Run Deployment Guide

## Deployment Files Overview

The following files have been created to ensure proper Cloud Run deployment:

### Core Deployment Files

1. **`Dockerfile`** - Container configuration for Cloud Run
   - Uses Python 3.11 slim base image
   - Installs required dependencies (fastapi, uvicorn, pydantic, flake8)
   - Includes curl for health checks
   - Configures proper port binding

2. **`run.py`** - Production entry point
   - Handles PORT environment variable from Cloud Run
   - Production-optimized uvicorn configuration
   - Proper logging configuration

3. **`start.sh`** - Startup script
   - Environment detection and configuration
   - Ensures health check dependencies are available
   - Flexible port binding

4. **`service.yaml`** - Cloud Run service configuration
   - Defines resource limits and requests
   - Configures health checks (liveness and readiness probes)
   - Sets up proper scaling and timeout parameters

## Key Fixes Applied

### 1. Port Configuration
- **Issue**: Fixed hardcoded port 5000 to use Cloud Run's PORT environment variable
- **Solution**: Application now reads PORT env var, defaults to 8080 for Cloud Run

### 2. Health Check Configuration
- **Issue**: Health checks were failing because application wasn't responding correctly
- **Solution**: 
  - Added proper `/health` endpoint returning 200 OK
  - Configured Docker health checks
  - Added liveness and readiness probes in service.yaml

### 3. Host Binding
- **Issue**: Application may not have been binding to correct interface
- **Solution**: Explicitly set host to "0.0.0.0" for external access

### 4. Entry Point
- **Issue**: Cloud Run couldn't find proper application entry point
- **Solution**: Created dedicated `run.py` entry point with production configuration

## Deployment Process

### Using Replit Deploy
1. Click the "Deploy" button in Replit
2. Select "Cloud Run" as deployment target
3. The system will automatically use the Dockerfile and configuration

### Manual Cloud Run Deployment
```bash
# Build container
docker build -t codeguard-api .

# Tag for Google Container Registry
docker tag codeguard-api gcr.io/YOUR_PROJECT_ID/codeguard-api

# Push to registry
docker push gcr.io/YOUR_PROJECT_ID/codeguard-api

# Deploy to Cloud Run
gcloud run deploy codeguard-api \
  --image gcr.io/YOUR_PROJECT_ID/codeguard-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 1 \
  --memory 512Mi \
  --timeout 300s \
  --max-instances 100
```

## Environment Variables

### Required for Production
- `PORT` - Set automatically by Cloud Run (usually 8080)
- `ENVIRONMENT` - Set to "production" for Cloud Run

### Optional for Authentication
- `CODEGUARD_API_KEY` - Set this for production API key authentication
  - If not set, runs in development mode (allows requests without auth)
  - If set, requires Bearer token authentication

## Health Check Endpoints

The application provides several endpoints for monitoring:

- `GET /health` - Basic health check (returns 200 OK)
- `GET /` - Service information and available endpoints
- `GET /auth/status` - Authentication status check

## Post-Deployment Verification

After deployment, verify the service works correctly:

```bash
# Test health endpoint
curl https://your-service-url/health

# Test root endpoint
curl https://your-service-url/

# Test audit endpoint (development mode)
curl -X POST https://your-service-url/audit \
  -H "Content-Type: application/json" \
  -d '{"files":[{"filename":"test.py","content":"import os\nprint(\"hello\")"}]}'
```

## Troubleshooting

### Common Issues

1. **503 Service Unavailable**
   - Check container logs for startup errors
   - Verify PORT environment variable is being read correctly
   - Ensure application starts within Cloud Run timeout

2. **Health Check Failures**
   - Verify `/health` endpoint returns 200 OK
   - Check that application binds to 0.0.0.0:$PORT
   - Ensure curl is installed in container for health checks

3. **Authentication Issues**
   - For development: Ensure no CODEGUARD_API_KEY is set
   - For production: Set CODEGUARD_API_KEY environment variable
   - Check auth logs for specific error details

### Monitoring
- Use Cloud Run logging to monitor application health
- Set up Cloud Monitoring alerts for 5xx errors
- Monitor response times and resource usage

## Security Considerations

- API runs in development mode by default (no auth required)
- For production, set CODEGUARD_API_KEY environment variable
- Consider setting up custom domain with TLS
- Implement rate limiting if needed for high traffic

The deployment configuration ensures Cloud Run health checks pass and the application responds correctly to all endpoints.