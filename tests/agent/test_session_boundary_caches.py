from __future__ import annotations

from hermes_constants import get_hermes_home


def test_clear_session_boundary_caches_resets_skill_env_and_credential_registries():
    from agent.session_boundary_caches import clear_session_boundary_caches
    from tools.credential_files import (
        clear_credential_files,
        get_credential_file_mounts,
        register_credential_file,
    )
    from tools.env_passthrough import (
        clear_env_passthrough,
        is_env_passthrough,
        register_env_passthrough,
    )

    env_name = "HERMES_TEST_PARITY_BOUNDARY_ENV"
    cred_rel = "parity-boundary-credential.json"
    cred_path = get_hermes_home() / cred_rel
    cred_path.parent.mkdir(parents=True, exist_ok=True)
    cred_path.write_text("{}\n", encoding="utf-8")

    try:
        clear_env_passthrough()
        clear_credential_files()
        register_env_passthrough([env_name])
        assert register_credential_file(cred_rel) is True

        assert is_env_passthrough(env_name) is True
        assert any(mount["container_path"].endswith(cred_rel) for mount in get_credential_file_mounts())

        clear_session_boundary_caches()

        assert is_env_passthrough(env_name) is False
        assert not any(mount["container_path"].endswith(cred_rel) for mount in get_credential_file_mounts())
    finally:
        clear_env_passthrough()
        clear_credential_files()


def test_clear_session_boundary_caches_preserves_config_backed_passthrough(monkeypatch):
    from agent.session_boundary_caches import clear_session_boundary_caches
    from tools.credential_files import (
        clear_credential_files,
        get_credential_file_mounts,
        register_credential_file,
    )
    from tools.env_passthrough import (
        clear_env_passthrough,
        is_env_passthrough,
        register_env_passthrough,
    )
    import tools.credential_files as credential_files_mod
    import tools.env_passthrough as env_passthrough_mod

    skill_env = "HERMES_TEST_BOUNDARY_SKILL_ENV"
    config_env = "HERMES_TEST_BOUNDARY_CONFIG_ENV"
    skill_rel = "parity-boundary-skill-credential.json"
    config_rel = "parity-boundary-config-credential.json"
    hermes_home = get_hermes_home()
    skill_path = hermes_home / skill_rel
    config_path = hermes_home / config_rel
    skill_path.write_text("{}\n", encoding="utf-8")
    config_path.write_text("{}\n", encoding="utf-8")
    from hermes_constants import hermes_home_key
    monkeypatch.setattr(
        env_passthrough_mod, "_config_passthrough", {hermes_home_key(): frozenset({config_env})}
    )
    monkeypatch.setattr(
        credential_files_mod,
        "_config_files",
        {hermes_home_key(): [{
            "host_path": str(config_path),
            "container_path": f"/root/.hermes/{config_rel}",
        }]},
    )

    try:
        clear_env_passthrough()
        clear_credential_files()
        register_env_passthrough([skill_env])
        assert register_credential_file(skill_rel) is True
        assert is_env_passthrough(skill_env) is True
        assert is_env_passthrough(config_env) is True
        assert any(mount["container_path"].endswith(skill_rel) for mount in get_credential_file_mounts())
        assert any(mount["container_path"].endswith(config_rel) for mount in get_credential_file_mounts())

        clear_session_boundary_caches()

        assert is_env_passthrough(skill_env) is False
        assert is_env_passthrough(config_env) is True
        assert not any(mount["container_path"].endswith(skill_rel) for mount in get_credential_file_mounts())
        assert any(mount["container_path"].endswith(config_rel) for mount in get_credential_file_mounts())
    finally:
        clear_env_passthrough()
        clear_credential_files()


def test_clear_session_boundary_caches_continues_when_one_registry_fails(monkeypatch):
    from agent.session_boundary_caches import clear_session_boundary_caches
    from tools.credential_files import (
        clear_credential_files,
        get_credential_file_mounts,
        register_credential_file,
    )
    from tools.env_passthrough import (
        clear_env_passthrough,
        is_env_passthrough,
        register_env_passthrough,
    )
    import tools.env_passthrough as env_passthrough_mod

    env_name = "HERMES_TEST_PARITY_BOUNDARY_FAIL_OPEN_ENV"
    cred_rel = "parity-boundary-fail-open-credential.json"
    cred_path = get_hermes_home() / cred_rel
    cred_path.parent.mkdir(parents=True, exist_ok=True)
    cred_path.write_text("{}\n", encoding="utf-8")

    def _raise_clear_failure() -> None:
        raise RuntimeError("synthetic env clear failure")

    try:
        clear_env_passthrough()
        clear_credential_files()
        register_env_passthrough([env_name])
        assert register_credential_file(cred_rel) is True
        monkeypatch.setattr(env_passthrough_mod, "clear_env_passthrough", _raise_clear_failure)

        clear_session_boundary_caches()

        # The failing registry remains uncleared, but later registries still run.
        assert is_env_passthrough(env_name) is True
        assert not any(mount["container_path"].endswith(cred_rel) for mount in get_credential_file_mounts())
    finally:
        clear_env_passthrough()
        clear_credential_files()
