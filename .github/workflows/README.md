# GitHub Actions CI/CD Workflows

This directory contains automated workflows for continuous integration and deployment.

## Workflows

### 1. CI Pipeline (`ci.yml`)
**Triggers:** Push to main/develop/feature branches, Pull Requests

**Jobs:**
- **Lint**: Code quality checks with flake8 and black
- **Test**: Run all unit tests across Python 3.10 and 3.11
- **Security**: Security vulnerability scanning with bandit
- **Docker Build**: Test Docker image build
- **Streamlit Test**: Validate dashboard syntax
- **Summary**: Aggregate results

**Status Badge:**
```markdown
![CI Pipeline](https://github.com/Humanitariansai/Cyber-Physical-Systems/workflows/CI%20Pipeline/badge.svg)
```

### 2. CD Pipeline (`cd.yml`)
**Triggers:** Push to main branch, Version tags (v*)

**Jobs:**
- **Deploy Staging**: Deploy to staging environment on main branch
- **Build Docker**: Build and push Docker images for tagged releases
- **Release**: Create GitHub releases for version tags

### 3. Weekly Health Check (`weekly-health-check.yml`)
**Triggers:** Every Monday at 9 AM UTC, Manual dispatch

**Jobs:**
- Run comprehensive test suite
- Check for outdated dependencies
- Security audit with safety
- Generate health report

## Setup Instructions

### 1. Enable GitHub Actions
GitHub Actions should be automatically enabled for your repository.

### 2. Add Secrets (Optional)
For Docker Hub integration, add these secrets in Repository Settings > Secrets:
- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub password/token

### 3. Status Badges
Add these badges to your README.md:

```markdown
![CI Pipeline](https://github.com/Humanitariansai/Cyber-Physical-Systems/workflows/CI%20Pipeline/badge.svg)
![CD Pipeline](https://github.com/Humanitariansai/Cyber-Physical-Systems/workflows/CD%20Pipeline/badge.svg)
![Weekly Health Check](https://github.com/Humanitariansai/Cyber-Physical-Systems/workflows/Weekly%20Health%20Check/badge.svg)
```

## Local Testing

Test workflows locally using [act](https://github.com/nektos/act):

```bash
# Install act
# Windows: choco install act-cli
# Mac: brew install act
# Linux: See act documentation

# Run CI pipeline locally
act -j test

# Run specific workflow
act -W .github/workflows/ci.yml
```

## Customization

### Modify Python Versions
Edit the matrix in `ci.yml`:
```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
```

### Add New Test Jobs
Add a new job in `ci.yml`:
```yaml
  my-new-test:
    name: My New Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run my test
        run: |
          pytest my_test.py
```

### Change Schedule
Modify the cron schedule in `weekly-health-check.yml`:
```yaml
schedule:
  # Run every day at 2 AM UTC
  - cron: '0 2 * * *'
```

## Troubleshooting

### Tests Failing
1. Check the workflow run logs in Actions tab
2. Run tests locally: `pytest -v`
3. Ensure all dependencies in requirements.txt

### Docker Build Failing
1. Verify Dockerfile exists and is valid
2. Test locally: `docker build -t test .`
3. Check Docker secrets are set correctly

### Security Scan Issues
Review the bandit report artifact for details on security concerns.

## Best Practices

1. **Keep workflows fast**: Use caching for dependencies
2. **Use continue-on-error**: For non-critical checks
3. **Add status checks**: Require CI to pass before merging PRs
4. **Regular updates**: Keep actions versions up to date
5. **Monitor runs**: Review failed workflows promptly

## Contributing

When adding new workflows:
1. Test locally with act
2. Use descriptive job names
3. Add comments for complex steps
4. Update this README
5. Follow YAML best practices
