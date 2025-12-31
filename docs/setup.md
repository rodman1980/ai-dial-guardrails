---
title: Setup Guide
description: Environment configuration, dependency installation, and getting started
version: 1.0.0
last_updated: 2025-12-31
related: [README.md, architecture.md]
tags: [setup, installation, configuration]
---

# Setup Guide

Complete environment setup for the AI DIAL Guardrails project.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Installation](#detailed-installation)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Development Tools](#development-tools)

## Prerequisites

### Required

- **Python 3.11+**: Check with `python3 --version`
- **pip**: Python package manager (included with Python 3.11+)
- **EPAM VPN Access**: Required for DIAL API endpoint
- **DIAL API Key**: Obtain from EPAM support portal

### Recommended

- **Git**: For version control
- **VS Code**: Recommended IDE with Python extension
- **Terminal**: bash, zsh, or equivalent

## Quick Start

For experienced developers who want to get running immediately:

```bash
# Clone repository
git clone git@git.epam.com:your-org/ai-dial-guardrails.git
cd ai-dial-guardrails

# Create virtual environment
python3.11 -m venv dial_guardrails
source dial_guardrails/bin/activate  # On Windows: dial_guardrails\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for Presidio
python -m spacy download en_core_web_sm

# Configure API key
export DIAL_API_KEY='your-key-here'  # On Windows: set DIAL_API_KEY=your-key-here

# Verify installation
python -c "from tasks._constants import API_KEY; print('API Key configured' if API_KEY else 'API Key missing')"

# Run first task
python tasks/t_1/prompt_injection.py
```

If successful, you'll see the interactive prompt injection REPL. Type `quit` to exit.

## Detailed Installation

### Step 1: Clone Repository

```bash
# Via SSH (recommended)
git clone git@git.epam.com:your-org/ai-dial-guardrails.git

# Via HTTPS
git clone https://git.epam.com/your-org/ai-dial-guardrails.git

cd ai-dial-guardrails
```

### Step 2: Create Virtual Environment

**Why virtual environment?** Isolates project dependencies from system Python.

```bash
# Create virtual environment
python3.11 -m venv dial_guardrails

# Activate (macOS/Linux)
source dial_guardrails/bin/activate

# Activate (Windows PowerShell)
dial_guardrails\Scripts\Activate.ps1

# Activate (Windows CMD)
dial_guardrails\Scripts\activate.bat
```

**Verify activation**: Your prompt should show `(dial_guardrails)` prefix.

### Step 3: Install Python Dependencies

```bash
# Upgrade pip (optional but recommended)
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

**Expected output**:
```
Successfully installed langchain-community-0.4.1 langchain-openai-1.0.2 presidio-analyzer-2.2.360 presidio-anonymizer-2.2.360 ...
```

### Step 4: Install spaCy Language Model

Presidio requires a spaCy NLP model for entity recognition.

```bash
# Download English language model
python -m spacy download en_core_web_sm
```

**Expected output**:
```
✔ Download and installation successful
You can now load the package via spacy.load('en_core_web_sm')
```

### Step 5: Configure DIAL API Access

#### 5a. Obtain DIAL API Key

1. Connect to EPAM VPN
2. Navigate to: https://support.epam.com/ess?id=sc_cat_item&table=sc_cat_item&sys_id=910603f1c3789e907509583bb001310c
3. Follow instructions to generate API key
4. Copy the key (looks like: `dial_xxx...`)

#### 5b. Set Environment Variable

**Option 1: Session-only (temporary)**

```bash
# macOS/Linux
export DIAL_API_KEY='your-key-here'

# Windows PowerShell
$env:DIAL_API_KEY='your-key-here'

# Windows CMD
set DIAL_API_KEY=your-key-here
```

**Option 2: Persistent (recommended)**

**macOS/Linux (bash/zsh)**:
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export DIAL_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Windows (PowerShell)**:
```powershell
# Set user environment variable
[System.Environment]::SetEnvironmentVariable('DIAL_API_KEY', 'your-key-here', [System.EnvironmentVariableTarget]::User)
```

**Option 3: .env File (for development)**

Create `.env` in project root:
```bash
DIAL_API_KEY=your-key-here
```

Then install `python-dotenv`:
```bash
pip install python-dotenv
```

And load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

**⚠️ Security Note**: Never commit `.env` or API keys to version control.

## Configuration

### Project Structure Verification

After setup, your project should look like:

```
ai-dial-guardrails/
├── dial_guardrails/          # Virtual environment (ignored by git)
│   ├── bin/
│   ├── lib/
│   └── ...
├── tasks/
│   ├── __init__.py
│   ├── _constants.py         # Contains DIAL_URL and API_KEY
│   ├── PROMPT_INJECTIONS_TO_TEST.md
│   ├── t_1/
│   ├── t_2/
│   └── t_3/
├── docs/                     # Documentation (this file)
├── requirements.txt
└── README.md
```

### Configuration Files

#### tasks/_constants.py

```python
import os

DIAL_URL = 'https://ai-proxy.lab.epam.com'
API_KEY = os.getenv('DIAL_API_KEY', '')
```

**Key Points**:
- `DIAL_URL`: EPAM internal DIAL proxy endpoint (no changes needed)
- `API_KEY`: Loaded from `DIAL_API_KEY` environment variable
- Default empty string if not set (will fail at runtime)

#### requirements.txt

```
langchain-community>=0.4.1
langchain-openai>=1.0.2
presidio-analyzer>=2.2.360
presidio_anonymizer>=2.2.360
```

**Dependencies**:
- `langchain-community`: Core LangChain components
- `langchain-openai`: Azure OpenAI integration
- `presidio-analyzer`: PII detection engine
- `presidio-anonymizer`: PII redaction/anonymization

## Verification

### 1. Check Python Version

```bash
python --version
# Expected: Python 3.11.x or 3.12.x
```

### 2. Check Virtual Environment

```bash
which python
# Expected (macOS/Linux): /path/to/ai-dial-guardrails/dial_guardrails/bin/python
# Expected (Windows): C:\path\to\ai-dial-guardrails\dial_guardrails\Scripts\python.exe
```

### 3. Check Dependencies

```bash
pip list | grep -E "(langchain|presidio)"
# Expected output:
# langchain-community    0.4.1
# langchain-core         0.x.x
# langchain-openai       1.0.2
# presidio-analyzer      2.2.360
# presidio-anonymizer    2.2.360
```

### 4. Check API Key Configuration

```bash
python -c "from tasks._constants import API_KEY; print('✓ API Key configured' if API_KEY else '✗ API Key missing')"
```

**Expected**: `✓ API Key configured`

### 5. Check spaCy Model

```bash
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✓ spaCy model loaded')"
```

**Expected**: `✓ spaCy model loaded`

### 6. Run Smoke Test

```bash
python tasks/t_1/prompt_injection.py
```

**Expected Output**:
```
=== Prompt Injection Exploration ===
Phase: Initializing LLM client...
Phase: LLM client ready.
Phase: Conversation initialized.

Enter your query (or 'quit' to exit): 
```

Type a simple query like `What is Amanda's phone number?` and verify you get a response.

## Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError: No module named 'langchain_openai'`

**Cause**: Dependencies not installed or wrong Python interpreter  
**Solution**:
```bash
# Ensure virtual environment is activated
source dial_guardrails/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### Issue: `OSError: [E050] Can't find model 'en_core_web_sm'`

**Cause**: spaCy model not downloaded  
**Solution**:
```bash
python -m spacy download en_core_web_sm
```

#### Issue: `API_KEY is empty` or connection errors

**Cause**: `DIAL_API_KEY` environment variable not set  
**Solution**:
```bash
export DIAL_API_KEY='your-key-here'

# Verify
echo $DIAL_API_KEY
```

#### Issue: `Connection refused` or timeout errors

**Cause**: EPAM VPN not connected  
**Solution**:
1. Connect to EPAM VPN
2. Verify connection: `curl https://ai-proxy.lab.epam.com` (should not timeout)
3. Re-run script

#### Issue: `Invalid API key` or `401 Unauthorized`

**Cause**: API key expired or incorrect  
**Solution**:
1. Regenerate API key from EPAM support portal
2. Update environment variable
3. Restart terminal/IDE to reload environment

#### Issue: Virtual environment not activating (Windows)

**Cause**: Execution policy restriction  
**Solution**:
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
dial_guardrails\Scripts\Activate.ps1
```

### Debugging Tips

**Enable verbose logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Test API connectivity**:
```python
from tasks._constants import DIAL_URL, API_KEY
print(f"DIAL_URL: {DIAL_URL}")
print(f"API_KEY set: {bool(API_KEY)}")
```

**Check LangChain version**:
```bash
pip show langchain-openai
```

## Development Tools

### Recommended VS Code Extensions

- **Python** (ms-python.python): IntelliSense, debugging, linting
- **Pylance** (ms-python.vscode-pylance): Fast language server
- **Jupyter** (ms-toolsai.jupyter): Notebook support (optional)

### Optional Development Dependencies

```bash
# Install development tools
pip install pytest black flake8 mypy

# pytest: Testing framework
# black: Code formatter
# flake8: Linter
# mypy: Static type checker
```

### IDE Configuration

**VS Code settings.json**:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/dial_guardrails/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true
}
```

### Running Tasks from IDE

**VS Code launch.json** (for debugging):
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Task 1: Prompt Injection",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/tasks/t_1/prompt_injection.py",
      "console": "integratedTerminal",
      "env": {
        "DIAL_API_KEY": "${env:DIAL_API_KEY}"
      }
    },
    {
      "name": "Task 2: Input Validation",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/tasks/t_2/input_llm_based_validation.py",
      "console": "integratedTerminal",
      "env": {
        "DIAL_API_KEY": "${env:DIAL_API_KEY}"
      }
    }
  ]
}
```

### Environment Management

**Activate environment shortcut** (add to shell config):
```bash
# ~/.bashrc or ~/.zshrc
alias activate-dial='cd /path/to/ai-dial-guardrails && source dial_guardrails/bin/activate'
```

**Deactivate virtual environment**:
```bash
deactivate
```

**Recreate environment** (if corrupted):
```bash
rm -rf dial_guardrails
python3.11 -m venv dial_guardrails
source dial_guardrails/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Next Steps

After successful setup:

1. **Explore Attack Vectors**: Review [PROMPT_INJECTIONS_TO_TEST.md](../tasks/PROMPT_INJECTIONS_TO_TEST.md)
2. **Run Task 1**: `python tasks/t_1/prompt_injection.py`
3. **Understand Architecture**: Read [architecture.md](./architecture.md)
4. **Implement Guardrails**: Complete Task 2 and Task 3
5. **Review API Docs**: See [api.md](./api.md) for interface details

---

**Need Help?** See [README.md](./README.md#getting-help) for support resources.
