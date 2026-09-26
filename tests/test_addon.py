import json

from wyoming_rvc import addon


def test_apply_options_maps_to_env_and_keeps_explicit_env(tmp_path):
    path = tmp_path / "options.json"
    path.write_text(
        json.dumps({"device": "auto", "rvc_pitch": 3, "wyoming_streaming": True, "rvc_model_file": "", "x": None})
    )
    env = {"RVC_PITCH": "5", "MODELS_DIR": "/models"}  # the image sets MODELS_DIR
    assert addon.apply_options(env, path) is True
    assert env == {"RVC_PITCH": "5", "DEVICE": "auto", "WYOMING_STREAMING": "true", "MODELS_DIR": "/data"}


def test_apply_options_noop_outside_addon(tmp_path):
    env = {}
    assert addon.apply_options(env, tmp_path / "missing.json") is False
    assert env == {}


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_announce_wyoming_posts_discovery(monkeypatch):
    monkeypatch.setattr(addon.socket, "gethostname", lambda: "addon-host")
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"], seen["auth"] = request.full_url, request.headers["Authorization"]
        seen["body"] = json.loads(request.data)
        return _Response()

    assert addon.announce_wyoming(10200, {"SUPERVISOR_TOKEN": "t0k"}, fake_urlopen) is True
    assert seen == {
        "url": "http://supervisor/discovery",
        "auth": "Bearer t0k",
        "body": {"service": "wyoming", "config": {"uri": "tcp://addon-host:10200"}},
    }


def test_announce_wyoming_skipped_without_token_and_tolerates_errors():
    assert addon.announce_wyoming(10200, {}) is False

    def boom(*_args, **_kwargs):
        raise OSError("no supervisor")

    assert addon.announce_wyoming(10200, {"SUPERVISOR_TOKEN": "t"}, boom) is False
