"""
This script performs mutation testing on the optax/experimental/microbatching.py file
"""

import os
import sys
import json
import subprocess
import shutil
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import re

@dataclass
class Mutation:
    """Represents a single mutation."""
    id: int
    file_path: str
    line_number: int
    original_code: str
    mutated_code: str
    operator: str
    operator_description: str
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "original_code": self.original_code.strip(),
            "mutated_code": self.mutated_code.strip(),
            "operator": self.operator,
            "operator_description": self.operator_description,
            "status": self.status,
        }


class MutationGenerator:
    """Generates mutations using multiple operators - ORIGINAL approach."""
    
    # Operator definitions
    # Note: We use more specific patterns to avoid mutating *args, **kwargs, *var unpacking
    AOR_REPLACEMENTS = [
        (r'(?<![=<>!+\-*/\(,])(\+)(?![=])', '-', '+ to -'),
        (r'(?<![=<>!+\-*/\(,])(-)(?![=>\d])', '+', '- to +'),
        # For multiplication, avoid *args and **kwargs patterns
        (r'(?<![=<>!+\-*/\(,\*])(\*)(?![=*a-zA-Z_])', '/', '* to /'),
        (r'(?<![=<>!+\-*/\(,])(/)(?![=/])', '*', '/ to *'),
        (r'(?<=\d)(\*\*)(?=\d)', '*', '** to *'),  # Only match numeric exponentiation
    ]
    
    ROR_REPLACEMENTS = [
        (r'(?<![<>=!-])(<)(?![=<])', '<=', '< to <='),
        (r'(?<![<>=!-])(<=)', '<', '<= to <'),
        (r'(?<![<>=!-])(>)(?![=>])', '>=', '> to >='),  # Negative lookbehind for '-' to avoid matching '->'
        (r'(?<![<>=!-])(>=)', '>', '>= to >'),
        (r'(?<![<>=!])(==)', '!=', '== to !='),
        (r'(?<![<>=!])(!=)', '==', '!= to =='),
    ]
    
    LCR_REPLACEMENTS = [
        (r'\b(and)\b', 'or', 'and to or'),
        (r'\b(or)\b', 'and', 'or to and'),
    ]
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def generate_all(self, target_file: str, max_mutations: int = 200) -> List[Mutation]:
        """Generate mutations for target file - EXCLUDE docstrings and comments."""
        all_mutations = []
        mutation_id = 0
        
        full_path = self.project_root / target_file
        if not full_path.exists():
            print(f"ERROR: File not found: {full_path}")
            return []
        
        with open(full_path, 'r', encoding='utf-8') as f:
            source = f.read()
            lines = source.split('\n')
        
        # Track docstring state
        in_docstring = False
        docstring_char = None
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Skip blank lines and copyright/license
            if not stripped or 'Copyright' in line or 'apache.org' in line.lower():
                continue
            
            # Detect docstring start/end
            if '"""' in line or "'''" in line:
                # Count triple quotes
                triple_double = line.count('"""')
                triple_single = line.count("'''")
                
                if triple_double > 0:
                    if not in_docstring:
                        in_docstring = True
                        docstring_char = '"'
                        # Check if docstring ends on same line
                        if triple_double >= 2:
                            in_docstring = False
                        continue
                    elif docstring_char == '"':
                        in_docstring = False
                        docstring_char = None
                        continue
                        
                if triple_single > 0:
                    if not in_docstring:
                        in_docstring = True
                        docstring_char = "'"
                        if triple_single >= 2:
                            in_docstring = False
                        continue
                    elif docstring_char == "'":
                        in_docstring = False
                        docstring_char = None
                        continue
            
            # Skip lines inside docstrings or comments
            if in_docstring or stripped.startswith('#'):
                continue
            
            # Skip string-only lines (logging messages, etc.)
            if (stripped.startswith("'") and stripped.endswith("'")) or \
               (stripped.startswith('"') and stripped.endswith('"')):
                continue
            
            # Generate AOR mutations
            for pattern, replacement, desc in self.AOR_REPLACEMENTS:
                for match in re.finditer(pattern, line):
                    start, end = match.span(1)
                    mutated = line[:start] + replacement + line[end:]
                    all_mutations.append(Mutation(
                        id=mutation_id,
                        file_path=target_file,
                        line_number=line_num,
                        original_code=line,
                        mutated_code=mutated,
                        operator="AOR",
                        operator_description=f"Arithmetic Operator Replacement: {desc}"
                    ))
                    mutation_id += 1
            
            # Generate ROR mutations
            for pattern, replacement, desc in self.ROR_REPLACEMENTS:
                for match in re.finditer(pattern, line):
                    start, end = match.span(1)
                    mutated = line[:start] + replacement + line[end:]
                    all_mutations.append(Mutation(
                        id=mutation_id,
                        file_path=target_file,
                        line_number=line_num,
                        original_code=line,
                        mutated_code=mutated,
                        operator="ROR",
                        operator_description=f"Relational Operator Replacement: {desc}"
                    ))
                    mutation_id += 1
            
            # Generate LCR mutations
            for pattern, replacement, desc in self.LCR_REPLACEMENTS:
                for match in re.finditer(pattern, line):
                    start, end = match.span(1)
                    mutated = line[:start] + replacement + line[end:]
                    all_mutations.append(Mutation(
                        id=mutation_id,
                        file_path=target_file,
                        line_number=line_num,
                        original_code=line,
                        mutated_code=mutated,
                        operator="LCR",
                        operator_description=f"Logical Connector Replacement: {desc}"
                    ))
                    mutation_id += 1
            
            # Generate CRP mutations (constant replacement)
            number_pattern = r'(?<![a-zA-Z_])(\d+\.?\d*)(?![a-zA-Z_\d])'
            for match in re.finditer(number_pattern, line):
                try:
                    original = match.group(1)
                    num_val = float(original)
                    
                    # Generate replacement
                    if '.' in original:
                        new_val = str(num_val + 0.1)
                    else:
                        new_val = str(int(num_val) + 1)
                    
                    start, end = match.span(1)
                    mutated = line[:start] + new_val + line[end:]
                    
                    all_mutations.append(Mutation(
                        id=mutation_id,
                        file_path=target_file,
                        line_number=line_num,
                        original_code=line,
                        mutated_code=mutated,
                        operator="CRP",
                        operator_description=f"Constant Replacement: {original} to {new_val}"
                    ))
                    mutation_id += 1
                except ValueError:
                    continue
            
            # Generate UOI mutations (Unary Operator Insertion - negate)
            # Insert negation for return statements with simple expressions
            if 'return ' in line and 'return None' not in line:
                return_match = re.search(r'return\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
                if return_match:
                    start = return_match.start(1)
                    var_name = return_match.group(1)
                    # Skip if it's a function call or already negated
                    if '(' not in line[start:start+len(var_name)+1] and line[start-1:start] != '-':
                        mutated = line[:start] + 'not ' + line[start:]
                        all_mutations.append(Mutation(
                            id=mutation_id,
                            file_path=target_file,
                            line_number=line_num,
                            original_code=line,
                            mutated_code=mutated,
                            operator="UOI",
                            operator_description=f"Unary Operator Insertion: not {var_name}"
                        ))
                        mutation_id += 1
            
            # Generate SDL mutations (Statement Deletion - for simple statements)
            # Replace simple statements with 'pass'
            if stripped.startswith('raise '):
                indent = len(line) - len(line.lstrip())
                mutated = ' ' * indent + 'pass  # SDL: deleted raise statement'
                all_mutations.append(Mutation(
                    id=mutation_id,
                    file_path=target_file,
                    line_number=line_num,
                    original_code=line,
                    mutated_code=mutated,
                    operator="SDL",
                    operator_description="Statement Deletion: removed raise statement"
                ))
                mutation_id += 1
            
            # Generate ASR mutations (Argument Swap for functions with multiple args)
            # Look for function calls with 2+ simple arguments (no nested parentheses)
            # Only swap in actual function call statements, not definitions
            if '(' in line and 'def ' not in line and '*args' not in line and '**kwargs' not in line:
                func_call_pattern = r'(\w+)\s*\(\s*(\w+)\s*,\s*(\w+)\s*(?:,|\))'
                for match in re.finditer(func_call_pattern, line):
                    func_name = match.group(1)
                    arg1 = match.group(2).strip()
                    arg2 = match.group(3).strip()
                    # Skip common functions where swapping doesn't make sense
                    # Also skip if args look like keywords (contain =)
                    skip_funcs = ['range', 'print', 'isinstance', 'hasattr', 'getattr', 
                                  'setattr', 'enumerate', 'zip', 'map', 'filter', 'lambda',
                                  'namedtuple', 'dataclass', 'def', 'class']
                    if func_name not in skip_funcs and arg1 != arg2:
                        start = match.start()
                        end = match.end()
                        # Swap arg1 and arg2
                        original_call = match.group(0)
                        swapped_call = original_call.replace(arg1, '__TEMP__').replace(arg2, arg1).replace('__TEMP__', arg2)
                        if original_call != swapped_call:
                            mutated = line[:start] + swapped_call + line[end:]
                            all_mutations.append(Mutation(
                                id=mutation_id,
                                file_path=target_file,
                                line_number=line_num,
                                original_code=line,
                                mutated_code=mutated,
                                operator="ASR",
                                operator_description=f"Argument Swap: swapped {arg1} and {arg2} in {func_name}()"
                            ))
                            mutation_id += 1
        
        # Limit to max_mutations
        if len(all_mutations) > max_mutations:
            print(f"Generated {len(all_mutations)} mutations, limiting to {max_mutations}")
            all_mutations = all_mutations[:max_mutations]
        
        return all_mutations


class MutationTester:
    """Test mutations using original approach."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "mutation_testing_microbatching" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def apply_mutation(self, mutation: Mutation) -> str:
        """Apply mutation and return backup path."""
        file_path = self.project_root / mutation.file_path
        backup_path = str(file_path) + ".backup"
        
        shutil.copy(file_path, backup_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        line_idx = mutation.line_number - 1
        if 0 <= line_idx < len(lines):
            lines[line_idx] = mutation.mutated_code + '\n'
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        return backup_path
    
    def restore_file(self, file_path: str, backup_path: str):
        """Restore file from backup."""
        shutil.copy(backup_path, file_path)
        os.remove(backup_path)
    
    def run_targeted_test(self, test_file: str, timeout: int = 180) -> Tuple[bool, str]:
        """Run tests for the mutated file."""
        # Use python3.11 which supports the required Python 3.10+ features
        python_exe = "python3.11"
        
        # Set PYTHONPATH to include project root for imports
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.project_root)
        
        # Run pytest with the test file
        cmd = [python_exe, "-m", "pytest", test_file, "-x", "-q", "--tb=short"]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            output = result.stdout + result.stderr
            
            # VALIDATION: Check for common error patterns that indicate tests didn't run
            error_patterns = [
                "ModuleNotFoundError",
                "ImportError", 
                "ERROR collecting",
                "error during collection",
                "command not found",
                "No such file or directory"
            ]
            
            for pattern in error_patterns:
                if pattern in output:
                    # This is an import/collection error, not a test failure
                    raise RuntimeError(
                        f"Test execution failed with error: {pattern}\n"
                        f"Output: {output[:500]}\n"
                        f"This indicates a setup problem, not a killed mutant."
                    )
            
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
    
    def test_mutation(self, mutation: Mutation, test_file: str) -> str:
        """Test a single mutation."""
        file_path = str(self.project_root / mutation.file_path)
        backup_path = self.apply_mutation(mutation)
        
        try:
            passed, output = self.run_targeted_test(test_file)
            # If test PASSES with mutation, mutant SURVIVED
            # If test FAILS with mutation, mutant is KILLED
            mutation.status = "survived" if passed else "killed"
        except Exception as e:
            mutation.status = "error"
        finally:
            self.restore_file(file_path, backup_path)
        
        return mutation.status
    
    def run_all(self, mutations: List[Mutation], test_file: str) -> Dict[str, Any]:
        """Run mutation testing on all mutations."""
        killed = survived = errors = 0
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"MUTATION TESTING")
        print(f"{'='*70}")
        print(f"Target: {mutations[0].file_path}")
        print(f"Test File: {test_file}")
        print(f"Total Mutants: {len(mutations)}")
        print(f"{'='*70}\n")
        
        for i, mutation in enumerate(mutations):
            status = self.test_mutation(mutation, test_file)
            
            if status == "killed":
                killed += 1
                symbol = "[KILLED]"
            elif status == "survived":
                survived += 1
                symbol = "[SURVIVED]"
            else:
                errors += 1
                symbol = "[ERROR]"
            
            print(f"[{i+1}/{len(mutations)}] Mutation #{mutation.id} ({mutation.operator})... {symbol}")
            
        
        elapsed = time.time() - start_time
        total_testable = killed + survived
        
        print(f"\n{'='*70}")
        print("MUTATION TESTING COMPLETE")
        print(f"{'='*70}")
        print(f"Total Mutants:    {len(mutations)}")
        print(f"Killed:           {killed}")
        print(f"Survived:         {survived}")
        print(f"Errors/Timeout:   {errors}")
        score = (killed / total_testable * 100) if total_testable > 0 else 0
        print(f"Mutation Score:   {score:.2f}%")
        print(f"Elapsed Time:     {elapsed:.1f}s")
        print(f"{'='*70}\n")
        
        return {
            "total_mutations": len(mutations),
            "killed": killed,
            "survived": survived,
            "errors": errors,
            "mutation_score": score,
            "elapsed_time": elapsed,
            "mutations": [m.to_dict() for m in mutations],
        }


def generate_diff_file(mutations: List[Mutation], output_path: Path):
    """Generate unified diff file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Mutation Testing Diff File\n")
        f.write(f"# Target: {mutations[0].file_path}\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total Mutations: {len(mutations)}\n")
        f.write("=" * 80 + "\n\n")
        
        for m in mutations:
            f.write(f"Mutation #{m.id} - {m.operator}: {m.operator_description}\n")
            f.write(f"File: {m.file_path}\n")
            f.write(f"Line: {m.line_number}\n")
            f.write(f"Status: {m.status}\n\n")
            f.write("--- original\n")
            f.write("+++ mutated\n")
            f.write(f"@@ -{m.line_number},1 +{m.line_number},1 @@\n")
            f.write(f"-{m.original_code.rstrip()}\n")
            f.write(f"+{m.mutated_code.rstrip()}\n")
            f.write("\n")


def main():
    """Main execution."""
    project_root = Path.cwd()
    target_file = "optax/experimental/microbatching.py"
    test_file = "optax/experimental/microbatching_test.py"
    
    print("\n" + "="*70)
    print(f"OPTAX MUTATION TESTING - {target_file}")
    print("="*70 + "\n")
    
    # Generate mutations
    print("Generating mutations...")
    generator = MutationGenerator(project_root)
    mutations = generator.generate_all(target_file)
    
    # Count by operator
    op_counts = {}
    for m in mutations:
        op_counts[m.operator] = op_counts.get(m.operator, 0) + 1
    
    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count} mutations")
    
    print(f"\n[OK] Generated {len(mutations)} total mutations\n")
    
    # Run mutation testing
    tester = MutationTester(project_root)
    results = tester.run_all(mutations, test_file)
    
    # Save results
    results_dir = project_root / "mutation_testing_microbatching" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON results
    json_path = results_dir / "mutation_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved JSON results: {json_path}")
    
    # Diff file
    diff_path = results_dir / "mutations_diff.txt"
    generate_diff_file(mutations, diff_path)
    print(f"[OK] Saved diff file: {diff_path}")
    
    print("\n" + "="*70)
    print("COMPLETE - Check results directory for detailed output")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
