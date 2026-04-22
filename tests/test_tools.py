"""Tests for shared tool utilities."""

import os
import tempfile

from lightspeed_agentic.tools import (
    augment_system_prompt,
    build_gemini_tools,
    discover_openai_skills,
    execute_bash,
    execute_glob,
    execute_grep,
    execute_read,
    execute_write,
    parse_bash_restrictions,
    validate_bash_command,
)


class TestBashRestrictions:
    def test_no_bash_entries(self):
        allowed, patterns = parse_bash_restrictions(["Read", "Glob"])
        assert allowed is False
        assert patterns == []

    def test_unrestricted_bash(self):
        allowed, patterns = parse_bash_restrictions(["Bash", "Read"])
        assert allowed is True
        assert patterns is None

    def test_unrestricted_bash_wildcard(self):
        allowed, patterns = parse_bash_restrictions(["Bash(*)", "Read"])
        assert allowed is True
        assert patterns is None

    def test_restricted_bash(self):
        allowed, patterns = parse_bash_restrictions(["Bash(oc:*)", "Bash(kubectl:*)"])
        assert allowed is True
        assert patterns == ["oc", "kubectl"]

    def test_validate_unrestricted(self):
        assert validate_bash_command("anything goes", None) is True

    def test_validate_allowed_prefix(self):
        assert validate_bash_command("oc get pods", ["oc", "kubectl"]) is True
        assert validate_bash_command("kubectl apply -f x.yaml", ["oc", "kubectl"]) is True

    def test_validate_blocked(self):
        assert validate_bash_command("rm -rf /", ["oc", "kubectl"]) is False

    def test_validate_exact_match(self):
        assert validate_bash_command("oc", ["oc"]) is True


class TestExecutors:
    def test_execute_bash_simple(self):
        result = execute_bash("echo hello", "/tmp", None)
        assert "hello" in result

    def test_execute_bash_restricted(self):
        result = execute_bash("rm -rf /", "/tmp", ["oc"])
        assert "not allowed" in result

    def test_execute_bash_timeout(self):
        result = execute_bash("sleep 10", "/tmp", None, timeout=1)
        assert "timed out" in result

    def test_execute_read(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3")
            f.flush()
            result = execute_read(f.name)
            assert "1\tline1" in result
            assert "2\tline2" in result
            os.unlink(f.name)

    def test_execute_read_nonexistent(self):
        result = execute_read("/nonexistent/file.txt")
        assert "Error" in result

    def test_execute_read_with_offset_limit(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("a\nb\nc\nd\ne")
            f.flush()
            result = execute_read(f.name, offset=1, limit=2)
            assert "2\tb" in result
            assert "3\tc" in result
            assert "1\ta" not in result
            os.unlink(f.name)

    def test_execute_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "test.txt")
            result = execute_write(path, "hello world")
            assert "Wrote" in result
            assert os.path.exists(path)
            assert open(path).read() == "hello world"

    def test_execute_glob(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "a.txt"), "w").close()
            open(os.path.join(d, "b.py"), "w").close()
            result = execute_glob("*.txt", d)
            assert "a.txt" in result
            assert "b.py" not in result

    def test_execute_glob_no_matches(self):
        with tempfile.TemporaryDirectory() as d:
            result = execute_glob("*.xyz", d)
            assert result == "No files found"


class TestGeminiTools:
    def test_builds_bash_tool(self):
        tools = build_gemini_tools(["Bash"], "/tmp")
        assert len(tools) == 1
        assert tools[0].__name__ == "run_bash"
        assert "bash command" in (tools[0].__doc__ or "").lower()

    def test_builds_all_tools(self):
        tools = build_gemini_tools(["Bash", "Read", "Write", "Glob", "Grep"], "/tmp")
        names = [t.__name__ for t in tools]
        assert "run_bash" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "glob_files" in names
        assert "grep_files" in names

    def test_respects_allowed_tools(self):
        tools = build_gemini_tools(["Read", "Glob"], "/tmp")
        names = [t.__name__ for t in tools]
        assert "run_bash" not in names
        assert "read_file" in names
        assert "glob_files" in names

    def test_empty_allowed_tools(self):
        tools = build_gemini_tools([], "/tmp")
        assert tools == []


class TestAugmentSystemPrompt:
    def test_prepends_claude_md(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "CLAUDE.md"), "w").write("# Project\nDo stuff")
            result = augment_system_prompt("You are an agent.", d)
            assert result.startswith("## Project Settings\n# Project")
            assert "You are an agent." in result

    def test_no_claude_md(self):
        with tempfile.TemporaryDirectory() as d:
            result = augment_system_prompt("You are an agent.", d)
            assert result == "You are an agent."


class TestDiscoverOpenAISkills:
    def test_discovers_skills(self):
        with tempfile.TemporaryDirectory() as d:
            skill_dir = os.path.join(d, "skills", "my-skill")
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
                f.write("---\nname: my-skill\ndescription: Does things\n---\n# Body")
            skills = discover_openai_skills(d)
            assert len(skills) == 1
            assert skills[0]["name"] == "my-skill"
            assert skills[0]["description"] == "Does things"

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            skills = discover_openai_skills(d)
            assert skills == []
