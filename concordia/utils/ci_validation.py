# Copyright 2025 SoyGema - CI validation utilities
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CI validation utilities for pre-push testing.

This module provides automated testing and validation to catch issues before
pushing to remote repositories. It mimics GitHub Actions CI pipeline locally.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional

def check_virtual_environment() -> bool:
    """Check if we're in the correct virtual environment."""
    virtual_env = os.environ.get('VIRTUAL_ENV', '')
    if 'evolutionary_env' not in virtual_env:
        print("❌ Error: Not in evolutionary_env virtual environment!")
        print("   Run: source evolutionary_env/bin/activate")
        return False
    
    print("✅ Virtual environment: evolutionary_env")
    return True

def check_python_path() -> bool:
    """Ensure PYTHONPATH is set correctly."""
    python_path = os.environ.get('PYTHONPATH', '')
    if '.' not in python_path:
        print("⚠️  Warning: PYTHONPATH may not include current directory")
        print("   Consider running with: PYTHONPATH=. python ...")
        return False
    
    print("✅ PYTHONPATH configured")
    return True

def run_command(cmd: List[str], description: str, allow_failure: bool = False) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"🔄 {description}...")
    
    try:
        # Set PYTHONPATH for all commands
        env = os.environ.copy()
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = '.'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {description} passed")
            return True, result.stdout
        else:
            if allow_failure:
                print(f"⚠️  {description} failed (allowed)")
                return False, result.stderr
            else:
                print(f"❌ {description} failed")
                print(f"   Error: {result.stderr}")
                return False, result.stderr
                
    except subprocess.TimeoutExpired:
        print(f"❌ {description} timed out")
        return False, "Command timed out"
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False, str(e)

def run_import_test() -> bool:
    """Test basic imports work correctly."""
    success, output = run_command(
        ['python', '-c', 'import concordia; print("Import successful")'],
        "Basic import test"
    )
    return success

def run_pytest() -> bool:
    """Run the full test suite."""
    success, output = run_command(
        ['python', '-m', 'pytest', '--tb=short', '-x'],
        "Running pytest test suite"
    )
    return success

def run_pytype() -> bool:
    """Run pytype static analysis (allowed to fail)."""
    success, output = run_command(
        ['pytype', 'concordia/'],
        "Running pytype static analysis",
        allow_failure=True
    )
    return True  # Always return True since pytype failures are allowed

def test_evolutionary_simulation() -> bool:
    """Test the evolutionary simulation runs without errors."""
    success, output = run_command(
        ['python', 'examples/evolutionary_simulation.py'],
        "Testing evolutionary simulation"
    )
    return success

def run_upstream_fixes() -> bool:
    """Apply upstream import fixes before testing."""
    success, output = run_command(
        ['python', '-c', 'from concordia.utils.upstream_fixes import run_all_fixes; run_all_fixes()'],
        "Applying upstream import fixes"
    )
    return success

def validate_project_structure() -> bool:
    """Validate that key project files exist."""
    required_files = [
        'concordia/__init__.py',
        'examples/evolutionary_simulation.py',
        'concordia/typing/evolutionary.py',
        'concordia/utils/checkpointing.py',
        'CLAUDE.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ Project structure validation passed")
    return True

def full_validation(skip_tests: bool = False) -> bool:
    """Run complete validation pipeline.
    
    Args:
        skip_tests: If True, skip time-consuming tests
        
    Returns:
        True if all validations pass
    """
    print("🚀 Starting CI validation pipeline...")
    
    # Check environment setup
    if not check_virtual_environment():
        return False
    
    check_python_path()  # Warning only
    
    # Validate project structure
    if not validate_project_structure():
        return False
    
    # Apply upstream fixes first
    if not run_upstream_fixes():
        print("❌ Failed to apply upstream fixes")
        return False
    
    # Run basic import test
    if not run_import_test():
        return False
    
    if not skip_tests:
        # Run evolutionary simulation test
        if not test_evolutionary_simulation():
            return False
        
        # Run pytest
        if not run_pytest():
            return False
    
    # Run pytype (allowed to fail)
    run_pytype()
    
    print("\n✅ All CI validations passed! Safe to push.")
    return True

def quick_validation() -> bool:
    """Run quick validation without time-consuming tests."""
    return full_validation(skip_tests=True)

def main() -> None:
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CI validation checks')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick validation without full tests')
    parser.add_argument('--fix-only', action='store_true',
                       help='Only apply upstream fixes')
    
    args = parser.parse_args()
    
    try:
        if args.fix_only:
            success = run_upstream_fixes()
        elif args.quick:
            success = quick_validation()
        else:
            success = full_validation()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()