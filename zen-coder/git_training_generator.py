#!/usr/bin/env python3
"""
Zen-Coder Training Data Generator
Extracts coherent software engineering patterns from git history
Learns from real development work, including iterations and improvements
"""

import os
import json
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import difflib
import re
from collections import defaultdict
import pygit2
import ast
import tokenize
import io

@dataclass
class CodeChange:
    """Represents a coherent code change with context"""
    commit_hash: str
    timestamp: datetime
    author: str
    message: str
    files_changed: List[str]
    additions: int
    deletions: int
    diff: str
    before_code: str
    after_code: str
    is_improvement: bool
    is_bugfix: bool
    is_refactor: bool
    is_feature: bool
    coherence_score: float
    efficiency_score: float
    
@dataclass
class DevelopmentSession:
    """Groups related commits into coherent development sessions"""
    session_id: str
    start_time: datetime
    end_time: datetime
    commits: List[CodeChange]
    total_changes: int
    session_type: str  # feature, bugfix, refactor, experimentation
    quality_trajectory: List[float]  # How code quality evolved
    iteration_count: int  # Number of times same code was modified
    final_outcome: str  # success, reverted, partial, ongoing

@dataclass
class TrainingExample:
    """Training example for Zen-Coder"""
    instruction: str
    context: str
    response: str
    metadata: Dict
    quality_score: float
    
class GitTrainingGenerator:
    """Generate training data from git repositories"""
    
    def __init__(self, repo_paths: List[str], output_dir: str = "zen_coder_data"):
        self.repo_paths = repo_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Pattern recognition
        self.improvement_patterns = [
            r'fix\w*', r'improve\w*', r'optimize\w*', r'refactor\w*',
            r'clean\w*', r'update\w*', r'enhance\w*', r'better\w*'
        ]
        
        self.circular_patterns = [
            r'revert\w*', r'back to', r'undo\w*', r'restore\w*'
        ]
        
        # Track iterations (when same file is edited multiple times)
        self.file_edit_history = defaultdict(list)
        
        # Quality metrics
        self.quality_indicators = {
            'good': ['test', 'perf', 'optimize', 'clean', 'modular', 'type', 'safe'],
            'bad': ['hack', 'temp', 'todo', 'fixme', 'xxx', 'broken', 'wip']
        }
        
    def analyze_repositories(self) -> List[DevelopmentSession]:
        """Analyze all repositories and extract development sessions"""
        all_sessions = []
        
        for repo_path in self.repo_paths:
            print(f"Analyzing {repo_path}...")
            sessions = self.extract_sessions_from_repo(repo_path)
            all_sessions.extend(sessions)
            
        print(f"Found {len(all_sessions)} development sessions")
        return all_sessions
    
    def extract_sessions_from_repo(self, repo_path: str) -> List[DevelopmentSession]:
        """Extract coherent development sessions from a single repo"""
        repo = pygit2.Repository(repo_path)
        sessions = []
        
        # Get all commits
        commits = []
        for commit in repo.walk(repo.head.target, pygit2.GIT_SORT_TIME):
            commits.append(commit)
        
        # Group commits into sessions (within 2 hours of each other)
        current_session = []
        session_gap_hours = 2
        
        for i, commit in enumerate(commits):
            if not current_session:
                current_session.append(commit)
            else:
                time_diff = (current_session[-1].commit_time - commit.commit_time) / 3600
                if time_diff <= session_gap_hours:
                    current_session.append(commit)
                else:
                    # Process completed session
                    if len(current_session) >= 2:  # Meaningful session
                        session = self.process_session(current_session, repo)
                        if session:
                            sessions.append(session)
                    current_session = [commit]
        
        # Process final session
        if len(current_session) >= 2:
            session = self.process_session(current_session, repo)
            if session:
                sessions.append(session)
        
        return sessions
    
    def process_session(self, commits: List, repo) -> Optional[DevelopmentSession]:
        """Process a group of commits into a development session"""
        changes = []
        
        for commit in reversed(commits):  # Process in chronological order
            change = self.analyze_commit(commit, repo)
            if change:
                changes.append(change)
        
        if not changes:
            return None
        
        # Detect session type and quality trajectory
        session_type = self.detect_session_type(changes)
        quality_trajectory = self.calculate_quality_trajectory(changes)
        iteration_count = self.count_iterations(changes)
        
        # Create session ID
        session_id = hashlib.md5(
            f"{changes[0].commit_hash}_{changes[-1].commit_hash}".encode()
        ).hexdigest()[:8]
        
        return DevelopmentSession(
            session_id=session_id,
            start_time=changes[0].timestamp,
            end_time=changes[-1].timestamp,
            commits=changes,
            total_changes=sum(c.additions + c.deletions for c in changes),
            session_type=session_type,
            quality_trajectory=quality_trajectory,
            iteration_count=iteration_count,
            final_outcome=self.determine_outcome(changes)
        )
    
    def analyze_commit(self, commit, repo) -> Optional[CodeChange]:
        """Analyze a single commit"""
        try:
            # Get diff
            if commit.parents:
                diff = repo.diff(commit.parents[0], commit)
            else:
                diff = None
            
            if not diff:
                return None
            
            files_changed = []
            additions = 0
            deletions = 0
            diff_text = ""
            before_code = ""
            after_code = ""
            
            for patch in diff:
                if not patch.delta.new_file.path.endswith(('.py', '.js', '.ts', '.go')):
                    continue
                    
                files_changed.append(patch.delta.new_file.path)
                
                # Count changes
                for hunk in patch.hunks:
                    for line in hunk.lines:
                        if line.origin == '+':
                            additions += 1
                            after_code += line.content
                        elif line.origin == '-':
                            deletions += 1
                            before_code += line.content
                        
                diff_text += str(patch.text)
            
            if not files_changed:
                return None
            
            # Classify commit
            message_lower = commit.message.lower()
            is_improvement = any(re.search(p, message_lower) for p in self.improvement_patterns)
            is_bugfix = 'fix' in message_lower or 'bug' in message_lower
            is_refactor = 'refactor' in message_lower or 'clean' in message_lower
            is_feature = 'feat' in message_lower or 'add' in message_lower or 'implement' in message_lower
            
            # Calculate scores
            coherence_score = self.calculate_coherence(commit.message, diff_text)
            efficiency_score = self.calculate_efficiency(before_code, after_code)
            
            return CodeChange(
                commit_hash=str(commit.id)[:8],
                timestamp=datetime.fromtimestamp(commit.commit_time),
                author=commit.author.name,
                message=commit.message.strip(),
                files_changed=files_changed,
                additions=additions,
                deletions=deletions,
                diff=diff_text[:10000],  # Limit size
                before_code=before_code[:5000],
                after_code=after_code[:5000],
                is_improvement=is_improvement,
                is_bugfix=is_bugfix,
                is_refactor=is_refactor,
                is_feature=is_feature,
                coherence_score=coherence_score,
                efficiency_score=efficiency_score
            )
            
        except Exception as e:
            print(f"Error analyzing commit: {e}")
            return None
    
    def calculate_coherence(self, message: str, diff: str) -> float:
        """Calculate how coherent a change is (message matches changes)"""
        # Simple heuristic: check if keywords in message appear in diff
        message_tokens = set(re.findall(r'\w+', message.lower()))
        diff_tokens = set(re.findall(r'\w+', diff.lower()))
        
        if not message_tokens:
            return 0.5
        
        overlap = len(message_tokens & diff_tokens)
        return min(overlap / len(message_tokens), 1.0)
    
    def calculate_efficiency(self, before: str, after: str) -> float:
        """Calculate efficiency improvement"""
        if not before or not after:
            return 0.5
        
        # Heuristics for efficiency
        score = 0.5
        
        # Fewer lines is often better
        before_lines = before.count('\n')
        after_lines = after.count('\n')
        if after_lines < before_lines:
            score += 0.1
        
        # Look for performance improvements
        perf_keywords = ['cache', 'memo', 'optimize', 'fast', 'async', 'parallel']
        for keyword in perf_keywords:
            if keyword in after.lower() and keyword not in before.lower():
                score += 0.1
        
        # Look for better patterns
        if 'for' in before and 'map' in after:  # Functional style
            score += 0.1
        if 'nested' not in after and 'nested' in before:  # Reduced nesting
            score += 0.1
            
        return min(score, 1.0)
    
    def detect_session_type(self, changes: List[CodeChange]) -> str:
        """Detect the type of development session"""
        if any(c.is_bugfix for c in changes):
            return "bugfix"
        elif any(c.is_feature for c in changes):
            return "feature"
        elif any(c.is_refactor for c in changes):
            return "refactor"
        elif len(changes) >= 5:  # Many small changes
            return "experimentation"
        else:
            return "maintenance"
    
    def calculate_quality_trajectory(self, changes: List[CodeChange]) -> List[float]:
        """Track how code quality evolved during session"""
        trajectory = []
        cumulative_score = 0.5
        
        for change in changes:
            # Adjust score based on change
            if change.is_improvement:
                cumulative_score += 0.1
            if change.efficiency_score > 0.7:
                cumulative_score += 0.05
            if 'revert' in change.message.lower():
                cumulative_score -= 0.2
            
            cumulative_score = max(0.1, min(1.0, cumulative_score))
            trajectory.append(cumulative_score)
        
        return trajectory
    
    def count_iterations(self, changes: List[CodeChange]) -> int:
        """Count how many times same files were modified"""
        file_counts = defaultdict(int)
        for change in changes:
            for file in change.files_changed:
                file_counts[file] += 1
        
        return max(file_counts.values()) if file_counts else 1
    
    def determine_outcome(self, changes: List[CodeChange]) -> str:
        """Determine the final outcome of the session"""
        last_message = changes[-1].message.lower()
        
        if 'revert' in last_message:
            return "reverted"
        elif 'wip' in last_message or 'todo' in last_message:
            return "ongoing"
        elif changes[-1].quality_trajectory[-1] > 0.7:
            return "success"
        else:
            return "partial"
    
    def generate_training_examples(self, sessions: List[DevelopmentSession]) -> List[TrainingExample]:
        """Generate training examples from development sessions"""
        examples = []
        
        for session in sessions:
            # Generate different types of examples from each session
            
            # 1. Full session example (learn entire workflow)
            if session.session_type in ["feature", "bugfix"] and session.final_outcome == "success":
                example = self.create_session_example(session)
                if example:
                    examples.append(example)
            
            # 2. Individual improvement examples
            for change in session.commits:
                if change.is_improvement and change.efficiency_score > 0.6:
                    example = self.create_improvement_example(change)
                    if example:
                        examples.append(example)
            
            # 3. Iteration learning (learn from repeated edits)
            if session.iteration_count > 2:
                example = self.create_iteration_example(session)
                if example:
                    examples.append(example)
            
            # 4. Error correction (learn from reverts and fixes)
            if any('revert' in c.message.lower() for c in session.commits):
                example = self.create_error_correction_example(session)
                if example:
                    examples.append(example)
        
        print(f"Generated {len(examples)} training examples")
        return examples
    
    def create_session_example(self, session: DevelopmentSession) -> Optional[TrainingExample]:
        """Create example from entire development session"""
        if not session.commits:
            return None
        
        first_commit = session.commits[0]
        last_commit = session.commits[-1]
        
        instruction = f"Implement a {session.session_type}: {first_commit.message}"
        
        context = f"""Current code:
```
{first_commit.before_code[:1000]}
```

Session type: {session.session_type}
Total commits needed: {len(session.commits)}
Files to modify: {', '.join(set(f for c in session.commits for f in c.files_changed[:3]))}
"""
        
        # Build response showing the development process
        response_parts = []
        for i, commit in enumerate(session.commits):
            response_parts.append(f"Step {i+1}: {commit.message}")
            if commit.after_code:
                response_parts.append(f"```\n{commit.after_code[:500]}\n```")
        
        response = "\n\n".join(response_parts)
        
        return TrainingExample(
            instruction=instruction,
            context=context,
            response=response,
            metadata={
                "session_id": session.session_id,
                "session_type": session.session_type,
                "iteration_count": session.iteration_count,
                "quality_trajectory": session.quality_trajectory
            },
            quality_score=session.quality_trajectory[-1] if session.quality_trajectory else 0.5
        )
    
    def create_improvement_example(self, change: CodeChange) -> Optional[TrainingExample]:
        """Create example from code improvement"""
        if not change.before_code or not change.after_code:
            return None
        
        instruction = f"Improve this code: {change.message}"
        
        context = f"""Original code:
```
{change.before_code[:1000]}
```
"""
        
        response = f"""Improved code:
```
{change.after_code[:1000]}
```

Changes made:
- {change.message}
- Lines added: {change.additions}
- Lines removed: {change.deletions}
- Efficiency score: {change.efficiency_score:.2f}
"""
        
        return TrainingExample(
            instruction=instruction,
            context=context,
            response=response,
            metadata={
                "commit_hash": change.commit_hash,
                "is_improvement": change.is_improvement,
                "efficiency_score": change.efficiency_score
            },
            quality_score=change.efficiency_score
        )
    
    def create_iteration_example(self, session: DevelopmentSession) -> Optional[TrainingExample]:
        """Create example from iterative development"""
        # Find the most iterated file
        file_iterations = defaultdict(list)
        for change in session.commits:
            for file in change.files_changed:
                file_iterations[file].append(change)
        
        if not file_iterations:
            return None
        
        most_iterated = max(file_iterations.items(), key=lambda x: len(x[1]))
        file_path, changes = most_iterated
        
        instruction = f"Iteratively improve {file_path} through multiple refinements"
        
        context = f"""File: {file_path}
Initial state:
```
{changes[0].before_code[:500]}
```

This file was edited {len(changes)} times in this session.
"""
        
        response_parts = [f"Iteration process for {file_path}:"]
        for i, change in enumerate(changes):
            response_parts.append(f"\nIteration {i+1}: {change.message}")
            response_parts.append(f"Efficiency: {change.efficiency_score:.2f}")
            if i == len(changes) - 1:  # Show final result
                response_parts.append(f"Final code:\n```\n{change.after_code[:500]}\n```")
        
        return TrainingExample(
            instruction=instruction,
            context=context,
            response="\n".join(response_parts),
            metadata={
                "file_path": file_path,
                "iteration_count": len(changes),
                "session_id": session.session_id
            },
            quality_score=session.quality_trajectory[-1] if session.quality_trajectory else 0.5
        )
    
    def create_error_correction_example(self, session: DevelopmentSession) -> Optional[TrainingExample]:
        """Create example from error corrections and reverts"""
        # Find the revert commit
        revert_commit = None
        original_commit = None
        
        for i, change in enumerate(session.commits):
            if 'revert' in change.message.lower():
                revert_commit = change
                if i > 0:
                    original_commit = session.commits[i-1]
                break
        
        if not revert_commit:
            return None
        
        instruction = "Learn from this mistake and its correction"
        
        context = f"""A change was made and then reverted.

Original change: {original_commit.message if original_commit else 'Unknown'}
Revert reason: {revert_commit.message}

Problematic code:
```
{original_commit.after_code[:500] if original_commit else 'N/A'}
```
"""
        
        response = f"""Learning points:
1. The original change caused issues
2. Revert message: {revert_commit.message}
3. Better approach would be to test changes more thoroughly
4. Consider gradual rollout or feature flags

Corrected code:
```
{revert_commit.after_code[:500]}
```
"""
        
        return TrainingExample(
            instruction=instruction,
            context=context,
            response=response,
            metadata={
                "session_id": session.session_id,
                "is_revert": True,
                "original_commit": original_commit.commit_hash if original_commit else None
            },
            quality_score=0.3  # Lower score for reverted changes
        )
    
    def save_training_data(self, examples: List[TrainingExample]):
        """Save training examples to JSON files"""
        # Split into train/val
        split_idx = int(len(examples) * 0.9)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        # Save training set
        train_file = self.output_dir / "zen_coder_train.json"
        with open(train_file, 'w') as f:
            json.dump(
                [asdict(e) for e in train_examples],
                f,
                indent=2,
                default=str
            )
        print(f"Saved {len(train_examples)} training examples to {train_file}")
        
        # Save validation set
        val_file = self.output_dir / "zen_coder_val.json"
        with open(val_file, 'w') as f:
            json.dump(
                [asdict(e) for e in val_examples],
                f,
                indent=2,
                default=str
            )
        print(f"Saved {len(val_examples)} validation examples to {val_file}")
        
        # Save metadata
        metadata = {
            "total_examples": len(examples),
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "average_quality": sum(e.quality_score for e in examples) / len(examples),
            "repos_analyzed": self.repo_paths,
            "generation_time": datetime.now().isoformat()
        }
        
        meta_file = self.output_dir / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {meta_file}")


def main():
    """Generate Zen-Coder training data from ~/work repositories"""
    
    # Find all git repositories in ~/work
    work_dir = Path.home() / "work"
    repo_paths = []
    
    print(f"Scanning {work_dir} for git repositories...")
    for path in work_dir.rglob(".git"):
        if path.is_dir():
            repo_path = str(path.parent)
            # Filter for relevant repos
            if any(x in repo_path for x in ['zen', 'hanzo', 'lux', 'zoo']):
                repo_paths.append(repo_path)
                print(f"  Found: {repo_path}")
    
    if not repo_paths:
        print("No repositories found!")
        return
    
    print(f"\nFound {len(repo_paths)} repositories to analyze")
    
    # Generate training data
    generator = GitTrainingGenerator(repo_paths)
    
    # Analyze repositories
    sessions = generator.analyze_repositories()
    
    # Generate training examples
    examples = generator.generate_training_examples(sessions)
    
    # Filter high-quality examples
    quality_examples = [e for e in examples if e.quality_score > 0.6]
    print(f"Filtered to {len(quality_examples)} high-quality examples")
    
    # Save training data
    generator.save_training_data(quality_examples)
    
    print("\nTraining data generation complete!")
    print("Use zen_coder_data/*.json to fine-tune Zen-Coder")


if __name__ == "__main__":
    main()