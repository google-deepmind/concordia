# Copyright 2025 DeepMind Technologies Limited.
# Copyright 2025 SoyGema - Upstream synchronization utilities
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

"""Automated utilities to fix common upstream import issues.

This module addresses recurring issues when syncing with the upstream Concordia
repository that often has missing or incomplete __init__.py files in contrib modules.
The goal is to automate these fixes so developers can focus on creating new features
instead of constantly fixing upstream mistakes.
"""

import os
from pathlib import Path
import sys
from typing import List, Tuple

# License header template for DeepMind modules
DEEPMIND_LICENSE_HEADER = '''# Copyright 2025 DeepMind Technologies Limited.
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

"""Library of components contributed by users."""
'''

def ensure_contrib_init_files() -> int:
    """Ensure all contrib directories have proper __init__.py files.
    
    Returns:
        Number of files created or fixed.
    """
    contrib_dirs = [
        'concordia/contrib',
        'concordia/contrib/components', 
        'concordia/contrib/components/agent',
        'concordia/contrib/components/agent/deprecated',
        'concordia/contrib/components/game_master',
        'concordia/contrib/components/game_master/deprecated',
        'concordia/contrib/prefabs',
        'concordia/contrib/prefabs/game_master',
        'concordia/contrib/data',
        'concordia/contrib/data/questionnaires',
        'concordia/contrib/deprecated',
        'concordia/contrib/deprecated/environment',
        'concordia/contrib/deprecated/environment/scenes',
    ]
    
    fixed_count = 0
    for dir_path in contrib_dirs:
        path = Path(dir_path)
        if path.exists():
            init_file = path / '__init__.py'
            if not init_file.exists():
                print(f"✅ Creating missing {init_file}")
                create_basic_init_file(init_file)
                fixed_count += 1
            elif init_file.stat().st_size < 100:  # Very small file, likely empty
                content = init_file.read_text().strip()
                if not content or 'Copyright 2023' in content:
                    print(f"✅ Updating minimal/outdated {init_file}")
                    create_basic_init_file(init_file)
                    fixed_count += 1
    
    return fixed_count

def create_basic_init_file(init_path: Path) -> None:
    """Create __init__.py with proper structure and updated license."""
    init_path.write_text(DEEPMIND_LICENSE_HEADER)

def add_explicit_exports() -> List[str]:
    """Add __all__ exports to existing __init__.py files that need them.
    
    Returns:
        List of files that were updated.
    """
    updated_files = []
    
    # Game master components that need explicit exports
    gm_init = Path('concordia/contrib/components/game_master/__init__.py')
    if gm_init.exists() and _needs_all_exports(gm_init):
        print("✅ Adding __all__ exports to game_master components")
        if _add_all_exports_to_file(gm_init):
            updated_files.append(str(gm_init))
    
    # Agent components that need explicit exports  
    agent_init = Path('concordia/contrib/components/agent/__init__.py')
    if agent_init.exists() and _needs_all_exports(agent_init):
        print("✅ Adding __all__ exports to agent components")
        if _add_all_exports_to_file(agent_init):
            updated_files.append(str(agent_init))
    
    # Agent deprecated components
    agent_dep_init = Path('concordia/contrib/components/agent/deprecated/__init__.py')
    if agent_dep_init.exists() and _needs_all_exports(agent_dep_init):
        print("✅ Adding __all__ exports to deprecated agent components")
        if _add_all_exports_to_file(agent_dep_init):
            updated_files.append(str(agent_dep_init))
    
    return updated_files

def _needs_all_exports(init_file: Path) -> bool:
    """Check if file needs __all__ exports added."""
    content = init_file.read_text()
    return '__all__' not in content and 'from concordia.contrib' in content

def _add_all_exports_to_file(init_file: Path) -> bool:
    """Add __all__ exports to a specific file."""
    content = init_file.read_text()
    lines = content.split('\n')
    
    # Find the last import line and extract module names
    last_import_idx = -1
    modules = []
    
    for i, line in enumerate(lines):
        if line.strip().startswith('from concordia.contrib.') and ' import ' in line:
            last_import_idx = i
            module = line.split('import')[-1].strip()
            if module and not module.startswith('#'):
                modules.append(f"    '{module}',")
    
    if last_import_idx >= 0 and modules:
        # Insert __all__ after imports
        all_declaration = [
            '',
            '# Explicit exports for pytype',
            '__all__ = [',
        ] + modules + [
            ']'
        ]
        
        lines = lines[:last_import_idx+1] + all_declaration + lines[last_import_idx+1:]
        init_file.write_text('\n'.join(lines))
        return True
    
    return False

def check_and_fix_specific_issues() -> List[str]:
    """Check for and fix specific known issues.
    
    Returns:
        List of issues fixed.
    """
    issues_fixed = []
    
    # Check if spaceship_system import is missing
    gm_init = Path('concordia/contrib/components/game_master/__init__.py')
    spaceship_py = Path('concordia/contrib/components/game_master/spaceship_system.py')
    
    if gm_init.exists() and spaceship_py.exists():
        content = gm_init.read_text()
        if 'spaceship_system' not in content:
            print("✅ Adding missing spaceship_system import")
            _add_import_to_file(gm_init, 'spaceship_system')
            issues_fixed.append('spaceship_system import')
    
    # Check if marketplace import is missing
    if gm_init.exists():
        marketplace_py = Path('concordia/contrib/components/game_master/marketplace.py')
        if marketplace_py.exists():
            content = gm_init.read_text()
            if 'marketplace' not in content:
                print("✅ Adding missing marketplace import")
                _add_import_to_file(gm_init, 'marketplace')
                issues_fixed.append('marketplace import')
    
    # Check agent situation_representation_via_narrative
    agent_init = Path('concordia/contrib/components/agent/__init__.py')
    situation_py = Path('concordia/contrib/components/agent/situation_representation_via_narrative.py')
    
    if agent_init.exists() and situation_py.exists():
        content = agent_init.read_text()
        if 'situation_representation_via_narrative' not in content:
            print("✅ Adding missing situation_representation_via_narrative import")
            _add_import_to_file(agent_init, 'situation_representation_via_narrative')
            issues_fixed.append('situation_representation_via_narrative import')
    
    return issues_fixed

def _add_import_to_file(init_file: Path, module_name: str) -> None:
    """Add an import statement to an __init__.py file."""
    content = init_file.read_text()
    lines = content.split('\n')
    
    # Find the right place to insert the import
    base_path = str(init_file.parent).replace('/', '.').replace('concordia.', 'concordia/')
    import_line = f'from {base_path} import {module_name}'
    
    # Find where other similar imports are
    insert_idx = len(lines)
    for i, line in enumerate(lines):
        if line.strip().startswith(f'from {base_path} import'):
            insert_idx = i + 1
    
    lines.insert(insert_idx, import_line)
    init_file.write_text('\n'.join(lines))

def run_all_fixes() -> Tuple[int, List[str], List[str]]:
    """Run all upstream fixes and return summary.
    
    Returns:
        Tuple of (files_created_count, files_updated, issues_fixed)
    """
    print("🔧 Starting upstream import fixes...")
    
    # Change to project root if needed
    current_path = Path.cwd()
    if current_path.name in ['utils', 'concordia']:
        while current_path.name != 'concordia' or not (current_path / 'concordia' / '__init__.py').exists():
            current_path = current_path.parent
            if current_path == current_path.parent:  # Reached filesystem root
                break
        os.chdir(current_path)
    
    # Verify we're in the right directory
    if not Path('concordia/__init__.py').exists():
        raise RuntimeError("Not in project root directory. Please run from concordia project root.")
    
    # Apply all fixes
    files_created = ensure_contrib_init_files()
    files_updated = add_explicit_exports()
    issues_fixed = check_and_fix_specific_issues()
    
    return files_created, files_updated, issues_fixed

def main() -> None:
    """Main function for command-line usage."""
    try:
        files_created, files_updated, issues_fixed = run_all_fixes()
        
        total_changes = files_created + len(files_updated) + len(issues_fixed)
        
        if total_changes > 0:
            print(f"\n✅ Successfully applied upstream fixes:")
            print(f"  📁 {files_created} __init__.py files created")
            print(f"  📝 {len(files_updated)} files updated with exports")
            print(f"  🔧 {len(issues_fixed)} specific issues fixed")
            if issues_fixed:
                print(f"     - {', '.join(issues_fixed)}")
        else:
            print("✅ No upstream fixes needed - all modules properly configured!")
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()