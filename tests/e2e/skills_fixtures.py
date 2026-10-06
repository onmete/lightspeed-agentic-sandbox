"""E2E skills workspace helpers for batch Jobs."""

from __future__ import annotations

from pathlib import Path

from kubernetes.client import ApiException, CoreV1Api, V1ConfigMap, V1ObjectMeta  # type: ignore[import-untyped]

from tests.e2e.suite_setup import E2E_COMPONENT_LABEL, E2E_COMPONENT_VALUE, E2E_RUN_LABEL

SKILLS_SOURCE = Path(__file__).resolve().parent / "workspace" / "skills"
E2E_POD_OUTPUT_DIR = "/tmp/lightspeed-e2e-output"  # noqa: S108 — pod-local emptyDir mount path
E2E_POD_SKILLS_DIR = "/app/skills"
E2E_POD_SKILLS_SRC_DIR = "/mnt/e2e-skills-src"
E2E_POD_SKILLS_WORKDIR = "/app/skills/.agents"


def list_skill_dirs() -> list[Path]:
    """Return fixture skill dirs (operator ``paths`` entries → ``/app/skills/{basename}``)."""
    if not SKILLS_SOURCE.is_dir():
        return []
    return sorted(
        path for path in SKILLS_SOURCE.iterdir() if path.is_dir() and (path / "SKILL.md").is_file()
    )


def skill_configmap_name(session_id: str, skill_name: str) -> str:
    return f"e2e-skill-{skill_name}-{session_id}"


def _cm_key_from_rel(rel: str) -> str:
    """Encode a relative path as a valid ConfigMap data key."""
    if "__" in rel:
        msg = f"skill path must not contain '__': {rel}"
        raise ValueError(msg)
    return rel.replace("/", "__")


def _rel_from_cm_key(key: str) -> str:
    """Decode a ConfigMap data key back to a relative path."""
    return key.replace("__", "/")


def build_skill_configmap_data(skill_dir: Path) -> dict[str, str]:
    """Build ConfigMap data for one skill directory tree."""
    data: dict[str, str] = {}
    for path in skill_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(skill_dir).as_posix()
            data[_cm_key_from_rel(rel)] = path.read_text(encoding="utf-8")
    if not data:
        msg = f"no files under skill dir {skill_dir}"
        raise FileNotFoundError(msg)
    return data


def configmap_items_for_skill(skill_dir: Path) -> list[dict[str, str]]:
    """Return ConfigMap volume items for mounting one skill at ``/app/skills/{basename}``."""
    return [
        {"key": key, "path": _rel_from_cm_key(key)} for key in build_skill_configmap_data(skill_dir)
    ]


def skill_materialize_script() -> str:
    """Shell for the init container: dereference ConfigMap symlinks into ``/app/skills``.

    Kubernetes ConfigMap volume mounts expose files as symlinks under ``..data/``
    and also contain a ``..data`` symlink plus a timestamped directory. Copying
    those entries makes DeepAgents list paths its backend then rejects.
    Copy only skill entries and dereference them so ``SKILL.md`` is a regular
    file (OpenAI ``LocalDirLazySkillSource`` skips non-regular files).
    """
    return (
        "set -euo pipefail\n"
        f'mkdir -p "{E2E_POD_SKILLS_WORKDIR}"\n'
        f'for src in "{E2E_POD_SKILLS_SRC_DIR}"/*; do\n'
        '  name=$(basename "${src}")\n'
        f'  dest="{E2E_POD_SKILLS_DIR}/${{name}}"\n'
        '  mkdir -p "${dest}"\n'
        "  find \"${src}\" -mindepth 1 -maxdepth 1 ! -name '..*' "
        '-exec cp -aL {} "${dest}/" \\;\n'
        "done\n"
    )


def ensure_skill_configmaps(
    core_api: CoreV1Api,
    namespace: str,
    session_id: str,
) -> dict[str, str]:
    """Create or update one ConfigMap per skill; return basename → ConfigMap name."""
    skill_dirs = list_skill_dirs()
    if not skill_dirs:
        msg = f"no skills under {SKILLS_SOURCE}"
        raise FileNotFoundError(msg)

    labels = {
        E2E_RUN_LABEL: session_id,
        E2E_COMPONENT_LABEL: E2E_COMPONENT_VALUE,
    }
    result: dict[str, str] = {}
    for skill_dir in skill_dirs:
        name = skill_configmap_name(session_id, skill_dir.name)
        body = V1ConfigMap(
            metadata=V1ObjectMeta(name=name, labels=labels),
            data=build_skill_configmap_data(skill_dir),
        )
        try:
            core_api.create_namespaced_config_map(namespace, body)
        except ApiException as exc:
            if exc.status != 409:
                raise
            core_api.patch_namespaced_config_map(name, namespace, body)
        result[skill_dir.name] = name
    return result
