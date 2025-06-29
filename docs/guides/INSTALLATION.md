# Installation Guide

Complete setup guide for CodeGuard API development and deployment.

## Prerequisites

- Python 3.11 or higher
- Git
- PostgreSQL (optional, for telemetry features)
- Node.js 16+ (for VS Code extension development)

## Local Development Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd codeguard
```

### 2. Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 3. Environment Configuration
Create `.env` file:
```bash
# Optional: API key for production mode
CODEGUARD_API_KEY=your-secure-api-key

# Optional: AI provider keys
OPENAI_API_KEY=your-openai-key
DEEPSEEK_API_KEY=your-deepseek-key
GEMINI_API_KEY=your-gemini-key
CLAUDE_API_KEY=your-claude-key

# Optional: Database for telemetry
DATABASE_URL=postgresql://user:pass@localhost:5432/codeguard
```

### 4. Start Development Server
```bash
python main.py
```

Server runs at: http://localhost:5000

## Production Deployment

### Replit Deployment (Recommended)

1. Fork repository to Replit
2. Set environment variables in Secrets
3. Server auto-deploys at: https://your-repl.replit.app

### Docker Deployment

```bash
# Build image
docker build -t codeguard-api .

# Run with Docker Compose
docker-compose up -d
```

### Manual Server Deployment

```bash
# Install production dependencies
pip install -r requirements.txt

# Start with gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:5000
```

## VS Code Extension Setup

### Install Extension
1. Download `.vsix` file from releases
2. Install via: Extensions → Install from VSIX
3. Configure API endpoint and key in settings

### Development Setup
```bash
cd vscode-extension
npm install
npm run compile

# Package extension
npm run package
```

## Database Setup (Optional)

### PostgreSQL
```sql
CREATE DATABASE codeguard;
CREATE USER codeguard WITH PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE codeguard TO codeguard;
```

### Environment Variable
```bash
DATABASE_URL=postgresql://codeguard:your-password@localhost:5432/codeguard
```

## Testing Setup

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov httpx

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Configuration Options

### API Settings
- `PORT`: Server port (default: 5000)
- `ENVIRONMENT`: development/production
- `CODEGUARD_API_KEY`: Authentication key

### Analysis Settings
- `MAX_FILE_SIZE`: Maximum file size in bytes
- `ANALYSIS_TIMEOUT`: Timeout for analysis in seconds
- `ENABLE_TELEMETRY`: Enable usage tracking

### AI Integration
- `OPENAI_API_KEY`: OpenAI GPT integration
- `DEFAULT_AI_PROVIDER`: Default AI provider
- `AI_TIMEOUT`: AI request timeout

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Reinstall dependencies
pip install -e . --force-reinstall
```

**Port Already in Use**
```bash
# Change port
export PORT=8000
python main.py
```

**Database Connection Issues**
```bash
# Check PostgreSQL service
sudo systemctl status postgresql

# Test connection
psql -h localhost -U codeguard -d codeguard
```

### Development Tools

**Code Formatting**
```bash
black . --line-length=88
isort . --profile=black
```

**Linting**
```bash
flake8 . --max-line-length=88
pylint *.py
mypy . --ignore-missing-imports
```

**API Testing**
```bash
# Test basic endpoint
curl http://localhost:5000/

# Test audit endpoint
curl -X POST http://localhost:5000/audit \
  -H "Content-Type: application/json" \
  -d '{"files": [{"filename": "test.py", "content": "print(\"hello\")"}]}'
```

## Performance Optimization

### Production Settings
- Use multiple workers: `gunicorn -w 4`
- Enable HTTP/2 with nginx reverse proxy
- Configure PostgreSQL connection pooling
- Set up Redis for caching (optional)

### Memory Management
- Monitor memory usage with analysis tools
- Configure file cleanup intervals
- Set reasonable timeout limits

## Security Considerations

### API Security
- Use strong API keys (minimum 32 characters)
- Enable HTTPS in production
- Configure CORS properly
- Implement rate limiting

### Data Protection
- Never store analyzed code permanently
- Clean up temporary files immediately
- Use secure environment variable storage
- Regular security audits with `bandit`

## Monitoring

### Health Checks
- GET `/health` - Basic health status
- GET `/metrics/usage` - System metrics
- Monitor disk space for temporary files

### Logging
- Configure structured logging
- Monitor error rates
- Track response times
- Set up alerts for failures