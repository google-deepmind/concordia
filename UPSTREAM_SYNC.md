<!--
Copyright 2023 DeepMind Technologies Limited.
Copyright 2025 [SoyGema] - Modifications and additions with Claude Code

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 🔄 Automated Upstream Sync

This repository includes an automated workflow that synchronizes with the upstream [google-deepmind/concordia](https://github.com/google-deepmind/concordia) repository daily and tests the evolutionary simulation functionality.

## 🚀 How It Works

The automation is implemented via GitHub Actions in `.github/workflows/upstream-sync.yml` and:

1. **Daily Sync**: Runs every day at 2 AM UTC
2. **Change Detection**: Checks if there are new commits in the upstream repository  
3. **Automatic Merge**: Creates a new branch and merges upstream changes
4. **Testing**: Runs the evolutionary simulation to ensure compatibility
5. **PR Creation**: Automatically creates a pull request with the changes

## 🧪 What Gets Tested

When upstream changes are detected, the workflow:

- ✅ **Evolutionary Simulation Test**: Runs a minimal version (3 generations, 4 agents, 3 rounds)
- ✅ **Core Concordia Tests**: Runs utility tests to ensure compatibility
- ✅ **Measurement Validation**: Verifies all measurement channels are working
- ✅ **Type Safety**: Confirms the modular typing system still works

## 📋 Manual Triggering

You can manually trigger the sync workflow:

1. Go to the **Actions** tab in your repository
2. Select **"Sync with Upstream and Test Evolutionary Simulation"**
3. Click **"Run workflow"**

## 🔧 Configuration

### Workflow Settings

The workflow is configured in `.github/workflows/upstream-sync.yml`:

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:      # Manual trigger
```

### Test Configuration

The test uses a minimal configuration for CI speed:

```python
config = EvolutionConfig(
    pop_size=4,
    num_generations=3,  # Reduced for CI
    selection_method='topk', 
    top_k=2,
    mutation_rate=0.2,
    num_rounds=3  # Reduced for CI
)
```

## 📨 Pull Request Format

When changes are detected, the workflow creates a PR with:

- **Title**: `🔄 Sync with upstream (YYYY-MM-DD)`
- **Labels**: `automated`, `upstream-sync`, `evolutionary-tested`
- **Content**: Summary of changes, test results, and commit SHAs

## 🛠️ Troubleshooting

### Merge Conflicts

If merge conflicts occur, the workflow will:
1. Abort the automatic merge
2. Exit with an error 
3. Require manual intervention

To resolve:
1. Manually merge the upstream changes
2. Resolve conflicts
3. Test the evolutionary simulation
4. Create a manual PR

### Test Failures

If the evolutionary simulation test fails:
1. Check the workflow logs for error details
2. Run the test locally: `python concordia/testing/test_evolutionary_simulation.py`
3. Fix any compatibility issues
4. Re-run the workflow

### No Changes Detected

If the workflow reports "Repository is up to date":
- This is normal when there are no new upstream commits
- The workflow will automatically exit without creating a PR

## 🔍 Monitoring

You can monitor the sync workflow:

1. **Actions Tab**: View workflow runs and logs
2. **Pull Requests**: Review automatically created sync PRs
3. **Notifications**: GitHub will notify you of new PRs

## 🎯 Benefits

This automation ensures:

- 🔄 **Always Up-to-Date**: Your fork stays synchronized with upstream
- 🧪 **Continuous Testing**: Changes are automatically tested
- 🤖 **Zero Maintenance**: Runs automatically without intervention  
- 🔒 **Quality Assurance**: Only tested changes are proposed for merge
- 📈 **Development Velocity**: Focus on features, not maintenance

---

*The evolutionary simulation continues to evolve, just like your codebase! 🧬*