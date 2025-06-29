# Privacy Policy for CodeGuard API

**Effective Date:** June 27, 2025  
**Last Updated:** June 27, 2025

## Overview

CodeGuard API ("we," "our," or "us") provides static code analysis services for machine learning and reinforcement learning Python code. This Privacy Policy explains how we collect, use, and protect information when you use our API service, including through OpenAI GPT Actions.

## Information We Collect

### Code Content
- **Python source code** submitted for analysis through the `/audit` endpoint
- **File names** and metadata associated with submitted code
- **Analysis requests** including any optional configuration parameters

### Technical Information
- **API usage data** including request timestamps, response times, and endpoint usage
- **Authentication tokens** (API keys) for access control
- **IP addresses** and request headers for security and operational purposes
- **Error logs** and system performance metrics

### Information We Do NOT Collect
- Personal identifying information (names, emails, addresses)
- User account information or profiles
- Persistent user tracking or behavioral analytics
- Third-party data or integrations beyond the submitted code

## How We Use Information

### Primary Purpose
- **Code Analysis**: Process submitted Python code to identify syntax errors, style issues, and best practices violations
- **Response Generation**: Provide structured JSON responses with issues, fixes, and recommendations

### Secondary Purposes
- **Service Operation**: Monitor API performance, uptime, and system health
- **Security**: Prevent abuse, unauthorized access, and ensure service stability
- **Improvement**: Analyze usage patterns to enhance service quality (aggregated data only)

## Data Processing and Storage

### Processing Location
- Code analysis is performed on secure servers hosted by Replit
- Processing occurs in real-time with temporary file creation during analysis
- No persistent storage of submitted code content

### Data Retention
- **Code Content**: Deleted immediately after analysis completion
- **Request Logs**: Retained for 30 days for operational purposes
- **Authentication Logs**: Retained for 90 days for security purposes
- **System Metrics**: Aggregated data retained for 1 year for service improvement

### Data Security
- All API communications use HTTPS encryption
- API key authentication required for access to analysis endpoints
- Temporary files are securely deleted after processing
- Access logs are monitored for security incidents

## Data Sharing and Disclosure

### We Do NOT Share
- Submitted code content with any third parties
- Individual user data or usage patterns
- Authentication credentials or API keys

### Limited Disclosure
We may disclose information only in these circumstances:
- **Legal Requirements**: When required by law, court order, or government regulation
- **Security Incidents**: To investigate potential security breaches or abuse
- **Service Providers**: With trusted infrastructure providers (Replit) under strict confidentiality agreements

## OpenAI GPT Actions Integration

### How It Works
- OpenAI GPT Actions may call our API on behalf of users
- We receive only the code content and parameters specified in the GPT Action request
- We do not have access to broader OpenAI user data or conversation context
- Authentication is handled through API keys provided by the GPT Action configuration

### Data Flow
1. User interacts with GPT that has CodeGuard Action enabled
2. GPT sends code analysis request to our API
3. We analyze the code and return structured results
4. GPT uses our response to provide feedback to the user
5. We immediately delete the temporary code files

## User Rights and Choices

### Access and Control
- No persistent user accounts or profiles are maintained
- Users control what code content is submitted for analysis
- API keys can be rotated or revoked at any time

### Data Minimization
- We only process the minimum data necessary for code analysis
- Optional parameters can be omitted to reduce data sharing
- Users can limit the scope of code submitted for analysis

## International Data Transfers

- Our service is hosted in the United States through Replit infrastructure
- Data processing occurs within Replit's secure environment
- We comply with applicable data protection laws for international users

## Children's Privacy

- Our service is not directed at children under 13
- We do not knowingly collect information from children under 13
- If we become aware of such collection, we will delete the information immediately

## Changes to This Policy

- We may update this policy to reflect service changes or legal requirements
- Material changes will be posted with an updated effective date
- Continued use of the service constitutes acceptance of policy updates

## Contact Information

For privacy-related questions or concerns:

- **Service**: CodeGuard API
- **Contact**: Through the API service documentation
- **Response Time**: We aim to respond to privacy inquiries within 5 business days

## Compliance

This privacy policy is designed to comply with:
- General Data Protection Regulation (GDPR)
- California Consumer Privacy Act (CCPA)
- OpenAI's requirements for GPT Action integrations
- Industry best practices for API services

## Technical Safeguards

### Encryption
- All data transmission uses TLS 1.2 or higher
- API keys are securely hashed for comparison
- Temporary files are created with restricted permissions

### Access Controls
- Authentication required for all analysis endpoints
- Request rate limiting to prevent abuse
- Monitoring for unusual access patterns

### Data Lifecycle
1. **Receipt**: Code content received via HTTPS
2. **Processing**: Temporary files created for analysis
3. **Analysis**: Static analysis tools process the code
4. **Response**: Structured results returned to client
5. **Cleanup**: All temporary files immediately deleted

---

**This privacy policy is effective as of the date listed above and governs the use of CodeGuard API services.**