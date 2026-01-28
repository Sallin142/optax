"""
This script performs mutation testing on the optax/tree_utils/_tree_math.py file

Only uses 4 mutation operators:
- AOR: Arithmetic Operator Replacement
- ROR: Relational Operator Replacement
- CRP: Constant Replacement Operator
- LCR: Logical Connector Replacement

Generates exactly 100 deterministic mutations.
"""

import os
import sys
import json
import subprocess
import shutil
import time
import ast
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
    """Generates mutations using exactly 4 operators: AOR, ROR, CRP, LCR."""

    # Arithmetic Operator Replacement (AOR)
    # Patterns to match arithmetic operators in code context
    AOR_PATTERNS = [
        # Addition to subtraction
        (r'(?<![eE])(\s*)\+(\s*)(?![+=])', r'\1-\2', '+ to -'),
        # Addition to multiplication
        (r'(?<![eE])(\s*)\+(\s*)(?![+=])', r'\1*\2', '+ to *'),
        # Subtraction to addition
        (r'(?<![eE\d])(\s*)-(\s*)(?![-=>\d])', r'\1+\2', '- to +'),
        # Subtraction to multiplication
        (r'(?<![eE\d])(\s*)-(\s*)(?![-=>\d])', r'\1*\2', '- to *'),
        # Multiplication to division
        (r'(?<!\*)(\s*)\*(\s*)(?!\*)', r'\1/\2', '* to /'),
        # Multiplication to addition
        (r'(?<!\*)(\s*)\*(\s*)(?!\*)', r'\1+\2', '* to +'),
        # Division to multiplication
        (r'(?<!/)(\s*)/(\s*)(?![/=])', r'\1*\2', '/ to *'),
        # Division to subtraction
        (r'(?<!/)(\s*)/(\s*)(?![/=])', r'\1-\2', '/ to -'),
        # Exponentiation to multiplication
        (r'\*\*', '*', '** to *'),
        # Exponentiation to addition
        (r'\*\*', '+', '** to +'),
    ]

    # Relational Operator Replacement (ROR)
    # Extended patterns for more thorough mutation coverage
    ROR_PATTERNS = [
        # <= mutations
        (r'(?<![<>=!])<=(?!=)', '<', '<= to <'),
        (r'(?<![<>=!])<=(?!=)', '>=', '<= to >='),
        (r'(?<![<>=!])<=(?!=)', '>', '<= to >'),
        (r'(?<![<>=!])<=(?!=)', '==', '<= to =='),
        (r'(?<![<>=!])<=(?!=)', '!=', '<= to !='),
        # < mutations
        (r'(?<![<>=!])<(?![<=])', '<=', '< to <='),
        (r'(?<![<>=!])<(?![<=])', '>', '< to >'),
        (r'(?<![<>=!])<(?![<=])', '>=', '< to >='),
        (r'(?<![<>=!])<(?![<=])', '==', '< to =='),
        (r'(?<![<>=!])<(?![<=])', '!=', '< to !='),
        # >= mutations
        (r'(?<![<>=!])>=(?!=)', '>', '>= to >'),
        (r'(?<![<>=!])>=(?!=)', '<=', '>= to <='),
        (r'(?<![<>=!])>=(?!=)', '<', '>= to <'),
        (r'(?<![<>=!])>=(?!=)', '==', '>= to =='),
        (r'(?<![<>=!])>=(?!=)', '!=', '>= to !='),
        # > mutations
        (r'(?<![<>=!])>(?![>=])', '>=', '> to >='),
        (r'(?<![<>=!])>(?![>=])', '<', '> to <'),
        (r'(?<![<>=!])>(?![>=])', '<=', '> to <='),
        (r'(?<![<>=!])>(?![>=])', '==', '> to =='),
        (r'(?<![<>=!])>(?![>=])', '!=', '> to !='),
        # == mutations
        (r'(?<![<>=!])==(?!=)', '!=', '== to !='),
        (r'(?<![<>=!])==(?!=)', '<', '== to <'),
        (r'(?<![<>=!])==(?!=)', '<=', '== to <='),
        (r'(?<![<>=!])==(?!=)', '>', '== to >'),
        (r'(?<![<>=!])==(?!=)', '>=', '== to >='),
        # != mutations
        (r'(?<![<>=!])!=(?!=)', '==', '!= to =='),
        (r'(?<![<>=!])!=(?!=)', '<', '!= to <'),
        (r'(?<![<>=!])!=(?!=)', '<=', '!= to <='),
        (r'(?<![<>=!])!=(?!=)', '>', '!= to >'),
        (r'(?<![<>=!])!=(?!=)', '>=', '!= to >='),
    ]

    # Logical Connector Replacement (LCR)
    LCR_PATTERNS = [
        (r'\band\b', 'or', 'and to or'),
        (r'\bor\b', 'and', 'or to and'),
    ]

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def _is_comment_line(self, line: str) -> bool:
        """Check if line is a comment (starts with #)."""
        return line.strip().startswith('#')

    def _is_in_string(self, line: str, pos: int) -> bool:
        """Check if position is inside a string literal."""
        in_single = False
        in_double = False
        i = 0
        while i < pos and i < len(line):
            char = line[i]
            # Handle escape sequences
            if i > 0 and line[i-1] == '\\':
                i += 1
                continue
            if char == "'" and not in_double:
                in_single = not in_single
            elif char == '"' and not in_single:
                in_double = not in_double
            i += 1
        return in_single or in_double

    def _strip_comments(self, line: str) -> str:
        """Remove inline comments from line, respecting string literals."""
        in_string = False
        string_char = None
        i = 0
        while i < len(line):
            char = line[i]
            # Handle string literals
            if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            # Handle comment start (only outside strings)
            if char == '#' and not in_string:
                return line[:i]
            i += 1
        return line

    def _is_version_comparison(self, line: str) -> bool:
        """Check if line contains version compatibility comparisons."""
        version_patterns = [
            r'__version__',
            r'jax\.__version__',
        ]
        for pattern in version_patterns:
            if re.search(pattern, line):
                return True
        return False

    def _validates_syntax(self, source: str) -> bool:
        """Check if source code compiles without syntax errors."""
        try:
            ast.parse(source)
            return True
        except SyntaxError:
            return False

    def _has_star_args(self, line: str) -> bool:
        """Check if line contains *args, **kwargs, or *unpacking patterns."""
        # Check for function definitions with *args/**kwargs
        if re.search(r'def\s+\w+\s*\([^)]*\*', line):
            return True
        # Check for *unpacking in lists, tuples, function calls
        if re.search(r'[\[\(,]\s*\*[a-zA-Z_]', line):
            return True
        # Check for **kwargs unpacking
        if re.search(r'\*\*[a-zA-Z_]', line):
            return True
        return False

    def _has_scientific_notation(self, line: str) -> bool:
        """Check if line contains scientific notation."""
        return bool(re.search(r'\d+\.?\d*[eE][+-]?\d+', line))

    def _generate_aor_mutations(self, line: str, line_num: int, code_only: str,
                                 lines: List[str], target_file: str) -> List[Mutation]:
        """Generate Arithmetic Operator Replacement mutations."""
        mutations = []

        # Skip lines with *args/**kwargs or scientific notation
        if self._has_star_args(code_only) or self._has_scientific_notation(code_only):
            return mutations

        for pattern, replacement, desc in self.AOR_PATTERNS:
            # Find all matches to generate one mutation per match
            for match in re.finditer(pattern, code_only):
                # Skip if match is in a string
                if self._is_in_string(code_only, match.start()):
                    continue

                # Apply replacement at this specific position
                mutated_code = code_only[:match.start()] + re.sub(pattern, replacement, match.group(), count=1) + code_only[match.end():]

                if mutated_code != code_only:
                    # Preserve any trailing comment
                    comment_part = line[len(code_only):] if len(line) > len(code_only) else ''
                    mutated_line = mutated_code + comment_part

                    # Validate syntax
                    test_lines = lines.copy()
                    test_lines[line_num - 1] = mutated_line
                    if self._validates_syntax('\n'.join(test_lines)):
                        mutations.append(Mutation(
                            id=0,  # Will be assigned later
                            file_path=target_file,
                            line_number=line_num,
                            original_code=line,
                            mutated_code=mutated_line,
                            operator="AOR",
                            operator_description=f"Arithmetic Operator Replacement: {desc}"
                        ))
                break  # Only one mutation per pattern per line for determinism

        return mutations

    def _generate_ror_mutations(self, line: str, line_num: int, code_only: str,
                                 lines: List[str], target_file: str) -> List[Mutation]:
        """Generate Relational Operator Replacement mutations."""
        mutations = []

        for pattern, replacement, desc in self.ROR_PATTERNS:
            match = re.search(pattern, code_only)
            if match:
                # Skip if in string
                if self._is_in_string(code_only, match.start()):
                    continue

                mutated_code = re.sub(pattern, replacement, code_only, count=1)

                if mutated_code != code_only:
                    comment_part = line[len(code_only):] if len(line) > len(code_only) else ''
                    mutated_line = mutated_code + comment_part

                    test_lines = lines.copy()
                    test_lines[line_num - 1] = mutated_line
                    if self._validates_syntax('\n'.join(test_lines)):
                        mutations.append(Mutation(
                            id=0,
                            file_path=target_file,
                            line_number=line_num,
                            original_code=line,
                            mutated_code=mutated_line,
                            operator="ROR",
                            operator_description=f"Relational Operator Replacement: {desc}"
                        ))

        return mutations

    def _generate_lcr_mutations(self, line: str, line_num: int, code_only: str,
                                 lines: List[str], target_file: str) -> List[Mutation]:
        """Generate Logical Connector Replacement mutations."""
        mutations = []

        for pattern, replacement, desc in self.LCR_PATTERNS:
            match = re.search(pattern, code_only)
            if match:
                # Skip if in string
                if self._is_in_string(code_only, match.start()):
                    continue

                mutated_code = re.sub(pattern, replacement, code_only, count=1)

                if mutated_code != code_only:
                    comment_part = line[len(code_only):] if len(line) > len(code_only) else ''
                    mutated_line = mutated_code + comment_part

                    test_lines = lines.copy()
                    test_lines[line_num - 1] = mutated_line
                    if self._validates_syntax('\n'.join(test_lines)):
                        mutations.append(Mutation(
                            id=0,
                            file_path=target_file,
                            line_number=line_num,
                            original_code=line,
                            mutated_code=mutated_line,
                            operator="LCR",
                            operator_description=f"Logical Connector Replacement: {desc}"
                        ))

        return mutations

    def _generate_crp_mutations(self, line: str, line_num: int, code_only: str,
                                 lines: List[str], target_file: str) -> List[Mutation]:
        """Generate Constant Replacement mutations."""
        mutations = []

        # Pattern to match numeric constants (integers and floats)
        # Avoid matching numbers in scientific notation exponents, version strings, etc.
        number_pattern = r'(?<![a-zA-Z_eE])(-?\d+\.?\d*)(?![a-zA-Z_\d])'

        for match in re.finditer(number_pattern, code_only):
            # Skip if in string
            if self._is_in_string(code_only, match.start()):
                continue

            original = match.group(1)
            start, end = match.span(1)

            # Skip version-like patterns
            context_before = code_only[max(0, start-3):start]
            context_after = code_only[end:min(len(code_only), end+3)]
            if "'" in context_before or '"' in context_before:
                continue
            if "'" in context_after or '"' in context_after:
                continue

            # Skip scientific notation exponents
            if re.search(r'[eE]$', code_only[:start]):
                continue

            try:
                num_val = float(original)
            except ValueError:
                continue

            # Generate different CRP variants
            # Use a set to track unique replacement values and avoid duplicates
            seen_values = set()
            crp_variants = []

            is_float = '.' in original

            # Variant 1: Increment by 1
            if is_float:
                new_val = str(num_val + 1.0)
            else:
                new_val = str(int(num_val) + 1)
            if new_val not in seen_values:
                seen_values.add(new_val)
                crp_variants.append((new_val, f"{original} to {new_val} (increment)"))

            # Variant 2: Decrement by 1 (if not already 0)
            if num_val != 0:
                if is_float:
                    new_val = str(num_val - 1.0)
                else:
                    new_val = str(int(num_val) - 1)
                if new_val not in seen_values:
                    seen_values.add(new_val)
                    crp_variants.append((new_val, f"{original} to {new_val} (decrement)"))

            # Variant 3: Replace with 0 (if not already 0)
            if num_val != 0 and "0" not in seen_values:
                seen_values.add("0")
                crp_variants.append(("0", f"{original} to 0"))

            # Variant 4: Negate (if not 0)
            if num_val != 0:
                if is_float:
                    new_val = str(-num_val)
                else:
                    new_val = str(-int(num_val))
                if new_val not in seen_values:
                    seen_values.add(new_val)
                    crp_variants.append((new_val, f"{original} to {new_val} (negate)"))

            # Variant 5: Replace with 1 (if not already 1)
            if "1" not in seen_values:
                seen_values.add("1")
                crp_variants.append(("1", f"{original} to 1"))

            # Variant 6: Replace with 2 (for more coverage)
            if "2" not in seen_values:
                seen_values.add("2")
                crp_variants.append(("2", f"{original} to 2"))

            for new_val, desc in crp_variants:
                mutated_code = code_only[:start] + new_val + code_only[end:]
                comment_part = line[len(code_only):] if len(line) > len(code_only) else ''
                mutated_line = mutated_code + comment_part

                test_lines = lines.copy()
                test_lines[line_num - 1] = mutated_line
                if self._validates_syntax('\n'.join(test_lines)):
                    mutations.append(Mutation(
                        id=0,
                        file_path=target_file,
                        line_number=line_num,
                        original_code=line,
                        mutated_code=mutated_line,
                        operator="CRP",
                        operator_description=f"Constant Replacement: {desc}"
                    ))

        return mutations

    def generate_all(self, target_file: str, max_mutations: int = 100) -> List[Mutation]:
        """Generate exactly max_mutations mutations for target file.

        Mutations are deterministic - same source file produces same mutations.
        Only uses AOR, ROR, CRP, LCR operators.
        EXCLUDES docstrings, comments, and version compatibility comparisons.
        """
        all_mutations = []

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

            # Skip blank lines
            if not stripped:
                continue

            # Skip comment lines
            if stripped.startswith('#'):
                continue

            # Skip copyright/license lines
            if 'Copyright' in line or 'apache.org' in line.lower():
                continue

            # Detect docstring start/end
            if '"""' in line or "'''" in line:
                triple_double = line.count('"""')
                triple_single = line.count("'''")

                if triple_double > 0:
                    if not in_docstring:
                        in_docstring = True
                        docstring_char = '"'
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

            # Skip lines inside docstrings
            if in_docstring:
                continue

            # Skip import statements
            if stripped.startswith('import ') or stripped.startswith('from '):
                continue

            # Skip version compatibility comparisons
            if self._is_version_comparison(line):
                continue

            # Skip decorator lines
            if stripped.startswith('@'):
                continue

            # Strip inline comments for mutation - only mutate code portion
            code_only = self._strip_comments(line)

            # Generate mutations using each operator
            all_mutations.extend(self._generate_aor_mutations(line, line_num, code_only, lines, target_file))
            all_mutations.extend(self._generate_ror_mutations(line, line_num, code_only, lines, target_file))
            all_mutations.extend(self._generate_lcr_mutations(line, line_num, code_only, lines, target_file))
            all_mutations.extend(self._generate_crp_mutations(line, line_num, code_only, lines, target_file))

        # Sort mutations by (line_number, operator, description) for deterministic ordering
        all_mutations.sort(key=lambda m: (m.line_number, m.operator, m.operator_description))

        # Deduplicate mutations based on (line_number, original_code, mutated_code)
        seen = set()
        unique_mutations = []
        for m in all_mutations:
            key = (m.line_number, m.original_code.strip(), m.mutated_code.strip())
            if key not in seen:
                seen.add(key)
                unique_mutations.append(m)
        all_mutations = unique_mutations

        # Limit to exactly max_mutations (deterministic - always same first N)
        if len(all_mutations) > max_mutations:
            all_mutations = all_mutations[:max_mutations]

        # Re-assign IDs to ensure they're sequential from 0 to len-1
        for i, mutation in enumerate(all_mutations):
            mutation.id = i

        return all_mutations


class MutationTester:
    """Test mutations using original approach."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "mutation_testing_tree_math" / "results"
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

    def run_targeted_test(self, test_file: str, timeout: int = 120) -> Tuple[bool, str]:
        """Run tests for the mutated file."""
        import platform
        if platform.system() == "Windows":
            python_exe = self.project_root / ".venv" / "Scripts" / "python.exe"
        else:
            python_exe = self.project_root / ".venv" / "bin" / "python3"

        if not python_exe.exists():
            python_exe = sys.executable

        # Verify pytest is available
        check_cmd = f'"{python_exe}" -c "import pytest"'
        check_result = subprocess.run(
            check_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        if check_result.returncode != 0:
            raise RuntimeError(
                f"pytest not available in {python_exe}. "
                f"Error: {check_result.stderr}\n"
                f"Please install dependencies: pip install -e '.[test]' pytest"
            )

        cmd = f'"{python_exe}" -m pytest {test_file} -x -q --tb=no'

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = result.stdout + result.stderr

            # Check for common error patterns
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
        f.write(f"# Operators: AOR, ROR, CRP, LCR only\n")
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
    target_file = "optax/tree_utils/_tree_math.py"
    test_file = "optax/tree_utils/_tree_math_test.py"

    print("\n" + "="*70)
    print(f"OPTAX MUTATION TESTING - {target_file}")
    print("="*70 + "\n")

    # Generate mutations (exactly 100, deterministic)
    print("Generating mutations (deterministic, exactly 100)...")
    print("Operators: AOR, ROR, CRP, LCR only")
    generator = MutationGenerator(project_root)
    mutations = generator.generate_all(target_file, max_mutations=100)

    print(f"Generated {len(mutations)} mutations")

    # Count by operator
    op_counts = {}
    for m in mutations:
        op_counts[m.operator] = op_counts.get(m.operator, 0) + 1

    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count} mutations")

    # Run mutation testing
    tester = MutationTester(project_root)
    results = tester.run_all(mutations, test_file)

    # Save results
    results_dir = project_root / "mutation_testing_tree_math" / "results"
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
