# -*- coding: utf-8 -*-
"""DeepSeek Search 搜索引擎测试套件。

覆盖范围：
1. base_url / endpoint 归一化
2. ``after:`` 时间窗口下推与 128 字符上限
3. 结构化结果解析（summary / site_name / date 过滤 / max_results）
4. OpenAI 兼容错误体解析
5. 熔断：连续 3 次失败 -> 不可用
6. SearchService 注册与优先级
7. Config：单 Key 复用、DEEPSEEK_SEARCH_ENABLED 关闭、显式 Key 覆盖、模型推断 v4-flash
8. config_registry：字段归类为 data_source

全部用例都用 mock 拦截 HTTP，不需要真实 DeepSeek Key。

运行：
    python -m pytest tests/test_deepseek_search.py -v
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import search_service as search_service_module  # noqa: E402
from src.config import Config  # noqa: E402
from src.search_service import (  # noqa: E402
    AnspireSearchProvider,
    DeepSeekSearchProvider,
    SearchService,
    reset_search_service,
)


class _FakeResponse:
    """Minimal requests.Response stand-in."""

    def __init__(self, payload, status_code=200, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or ""
        self.headers = {"content-type": "application/json"}
        self.content = b"{}" if payload is not None else b""

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _item(title="标题", url="https://example.com/a", date=None, summary="", site_name=None):
    return {
        "title": title,
        "url": url,
        "snippet": "短摘要",
        "summary": summary,
        "content": "",
        "site_name": site_name,
        "date": date,
        "authority": 3,
        "rerank_score": 2.4,
    }


def _recent_iso(days_ago=0):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


# ---------------------------------------------------------------------------
# 1. endpoint 归一化
# ---------------------------------------------------------------------------

def test_endpoint_defaults_to_official_search_api():
    provider = DeepSeekSearchProvider(["sk-test"])
    assert provider.endpoint == "https://api.deepseek.com/v1/tools/search"
    assert provider.name == "DeepSeek"


def test_endpoint_accepts_v1_and_trailing_slash():
    for base in ("https://api.deepseek.com/", "https://api.deepseek.com/v1", "https://api.deepseek.com/v1/"):
        assert DeepSeekSearchProvider(["sk"], base_url=base).endpoint == (
            "https://api.deepseek.com/v1/tools/search"
        )


def test_endpoint_keeps_custom_gateway_path():
    provider = DeepSeekSearchProvider(["sk"], base_url="https://gateway.example.com/deepseek")
    assert provider.endpoint == "https://gateway.example.com/deepseek/v1/tools/search"


# ---------------------------------------------------------------------------
# 2. after: 时间下推
# ---------------------------------------------------------------------------

def test_query_pushes_recency_window_with_after_operator():
    provider = DeepSeekSearchProvider(["sk"])
    built = provider._build_query("贵州茅台 2025 年报", 7)
    assert built.startswith("贵州茅台 2025 年报 after:")
    cutoff = datetime.strptime(built.split("after:", 1)[1], "%Y-%m-%d")
    expected = datetime.now(timezone.utc) - timedelta(days=7)
    assert abs((cutoff.date() - expected.date()).days) <= 1


def test_query_does_not_duplicate_existing_after_operator():
    provider = DeepSeekSearchProvider(["sk"])
    built = provider._build_query("宁德时代 after:2025-01-01", 3)
    assert built.count("after:") == 1


def test_query_respects_128_char_limit():
    provider = DeepSeekSearchProvider(["sk"])
    long_query = "贵州茅台" * 40  # 160 chars
    built = provider._build_query(long_query, 7)
    assert len(built) <= 128
    assert "after:" not in built  # 放不下时跳过下推，交给客户端过滤


# ---------------------------------------------------------------------------
# 3. 结构化结果解析
# ---------------------------------------------------------------------------

def test_parses_structured_results_and_prefers_summary_and_site_name():
    provider = DeepSeekSearchProvider(["sk"])
    payload = {
        "results": [
            _item(title="茅台发布年报", date=_recent_iso(1), summary="权威摘要" * 10, site_name="证券时报"),
        ]
    }
    with patch.object(search_service_module, "_post_with_retry", return_value=_FakeResponse(payload)) as post:
        response = provider.search("贵州茅台 年报", max_results=5, days=7)

    assert response.success is True
    assert response.provider == "DeepSeek"
    assert len(response.results) == 1
    result = response.results[0]
    assert result.title == "茅台发布年报"
    assert result.url == "https://example.com/a"
    assert result.source == "证券时报"
    assert result.snippet.startswith("权威摘要")
    assert result.published_date == payload["results"][0]["date"]

    # 请求形状：endpoint / Bearer / count / need_content / after:
    args, kwargs = post.call_args
    assert args[0] == "https://api.deepseek.com/v1/tools/search"
    assert kwargs["headers"]["Authorization"] == "Bearer sk"
    assert kwargs["json"]["count"] == 5
    assert kwargs["json"]["need_content"] is False
    assert "after:" in kwargs["json"]["query"]


def test_falls_back_to_domain_when_site_name_missing():
    provider = DeepSeekSearchProvider(["sk"])
    payload = {"results": [_item(url="https://www.cls.cn/detail/1", date=None, site_name=None)]}
    with patch.object(search_service_module, "_post_with_retry", return_value=_FakeResponse(payload)):
        response = provider.search("测试", max_results=3, days=3)
    assert response.results[0].source == "cls.cn"


def test_drops_results_outside_time_window_and_caps_max_results():
    provider = DeepSeekSearchProvider(["sk"])
    payload = {
        "results": [
            _item(title="两千年前", date="2000-01-01T00:00:00+00:00"),
            _item(title="近期1", date=_recent_iso(0), url="https://a.example/1"),
            _item(title="近期2", date=_recent_iso(2), url="https://a.example/2"),
            _item(title="近期3", date=_recent_iso(3), url="https://a.example/3"),
        ]
    }
    with patch.object(search_service_module, "_post_with_retry", return_value=_FakeResponse(payload)):
        response = provider.search("测试", max_results=2, days=3)

    assert response.success is True
    assert [r.title for r in response.results] == ["近期1", "近期2"]


def test_need_content_flag_is_forwarded():
    provider = DeepSeekSearchProvider(["sk"], need_content=True)
    payload = {"results": [_item(date=_recent_iso(0))]}
    with patch.object(search_service_module, "_post_with_retry", return_value=_FakeResponse(payload)) as post:
        provider.search("测试", max_results=1, days=3)
    assert post.call_args[1]["json"]["need_content"] is True


# ---------------------------------------------------------------------------
# 4. 错误处理
# ---------------------------------------------------------------------------

def test_http_401_returns_structured_error_message():
    provider = DeepSeekSearchProvider(["bad-key"])
    payload = {"error": {"message": "Authentication Fails", "type": "invalid_request_error", "param": None}}
    with patch.object(
        search_service_module, "_post_with_retry", return_value=_FakeResponse(payload, status_code=401)
    ):
        response = provider.search("测试", max_results=3, days=3)

    assert response.success is False
    assert "401" in response.error_message
    assert "Authentication Fails" in response.error_message


def test_missing_results_array_is_reported_as_failure():
    provider = DeepSeekSearchProvider(["sk"])
    with patch.object(search_service_module, "_post_with_retry", return_value=_FakeResponse({"unexpected": []})):
        response = provider.search("测试", max_results=3, days=3)
    assert response.success is False
    assert "results" in response.error_message


def test_network_error_is_caught():
    provider = DeepSeekSearchProvider(["sk"])
    with patch.object(
        search_service_module,
        "_post_with_retry",
        side_effect=search_service_module.requests.exceptions.ConnectionError("boom"),
    ):
        response = provider.search("测试", max_results=3, days=3)
    assert response.success is False
    assert "Network error" in response.error_message


# ---------------------------------------------------------------------------
# 5. 熔断
# ---------------------------------------------------------------------------

def test_circuit_breaker_opens_after_three_failures_and_resets_on_success():
    provider = DeepSeekSearchProvider(["sk"])
    assert provider.is_available is True

    for _ in range(3):
        provider._record_error("sk")
    assert provider.is_available is False

    # 冷却期内不可用
    provider._circuit_open_until = 0.0  # 模拟冷却结束 -> half-open
    assert provider.is_available is True

    provider._record_success("sk")
    assert provider._consecutive_failures == 0


# ---------------------------------------------------------------------------
# 6. SearchService 注册与优先级
# ---------------------------------------------------------------------------

def test_search_service_registers_deepseek_first_by_default():
    service = SearchService(deepseek_keys=["sk-deepseek"], tavily_keys=["tvly-x"])
    assert isinstance(service._providers[0], DeepSeekSearchProvider)
    assert [p.name for p in service._providers] == ["DeepSeek", "Tavily"]


def test_anspire_still_outranks_deepseek_when_explicitly_configured():
    service = SearchService(deepseek_keys=["sk-deepseek"], anspire_keys=["anspire-key"])
    assert isinstance(service._providers[0], AnspireSearchProvider)
    assert isinstance(service._providers[1], DeepSeekSearchProvider)


def test_constructor_kwargs_round_trip_keeps_deepseek_config():
    service = SearchService(
        deepseek_keys=["sk"],
        deepseek_base_url="https://api.deepseek.com/v1",
        deepseek_need_content=True,
    )
    assert service._constructor_kwargs["deepseek_keys"] == ["sk"]
    assert service._constructor_kwargs["deepseek_base_url"] == "https://api.deepseek.com/v1"
    assert service._constructor_kwargs["deepseek_need_content"] is True


# ---------------------------------------------------------------------------
# 7. Config 环境变量
# ---------------------------------------------------------------------------

_ENV_KEYS_TO_CLEAR = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_API_KEYS",
    "DEEPSEEK_SEARCH_API_KEYS",
    "DEEPSEEK_SEARCH_ENABLED",
    "DEEPSEEK_SEARCH_BASE_URL",
    "DEEPSEEK_SEARCH_NEED_CONTENT",
    "LITELLM_MODEL",
    "LITELLM_FALLBACK_MODELS",
    "LLM_CHANNELS",
    "LITELLM_CONFIG",
    "GEMINI_API_KEY",
    "GEMINI_API_KEYS",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEYS",
    "OPENAI_API_KEY",
    "OPENAI_API_KEYS",
    "MODELFLARE_KEY",
    "AIHUBMIX_KEY",
    "ANSPIRE_API_KEYS",
    "TAVILY_API_KEYS",
    "BOCHA_API_KEYS",
    "BRAVE_API_KEYS",
    "SERPAPI_API_KEYS",
    "MINIMAX_API_KEYS",
    "SEARXNG_BASE_URLS",
)


def _build_config(monkeypatch, tmp_path, env):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("ENV_FILE", str(env_file))
    for key in _ENV_KEYS_TO_CLEAR:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    Config.reset_instance()
    reset_search_service()
    return Config.get_instance()


def test_single_deepseek_key_enables_search_and_infers_v4_flash(monkeypatch, tmp_path):
    config = _build_config(monkeypatch, tmp_path, {"DEEPSEEK_API_KEY": "sk-deepseek-123456"})
    try:
        assert config.deepseek_search_api_keys == ["sk-deepseek-123456"]
        assert config.deepseek_search_base_url == "https://api.deepseek.com"
        assert config.deepseek_search_need_content is False
        assert config.litellm_model == "deepseek/deepseek-v4-flash"
        assert config.has_search_capability_enabled() is True
    finally:
        Config.reset_instance()


def test_search_enabled_false_keeps_key_for_llm_only(monkeypatch, tmp_path):
    config = _build_config(
        monkeypatch,
        tmp_path,
        {"DEEPSEEK_API_KEY": "sk-deepseek-123456", "DEEPSEEK_SEARCH_ENABLED": "false"},
    )
    try:
        assert config.deepseek_search_api_keys == []
        assert config.has_search_capability_enabled() is False
    finally:
        Config.reset_instance()


def test_explicit_search_keys_override_llm_keys(monkeypatch, tmp_path):
    config = _build_config(
        monkeypatch,
        tmp_path,
        {
            "DEEPSEEK_API_KEYS": "sk-llm-1,sk-llm-2",
            "DEEPSEEK_SEARCH_API_KEYS": " sk-search-1 , sk-search-2 ",
            "DEEPSEEK_SEARCH_BASE_URL": "https://api.deepseek.com/v1/",
            "DEEPSEEK_SEARCH_NEED_CONTENT": "true",
        },
    )
    try:
        assert config.deepseek_search_api_keys == ["sk-search-1", "sk-search-2"]
        assert config.deepseek_search_base_url == "https://api.deepseek.com/v1"
        assert config.deepseek_search_need_content is True
        assert config.deepseek_api_keys == ["sk-llm-1", "sk-llm-2"]
        assert config.litellm_model == "deepseek/deepseek-v4-flash"
    finally:
        Config.reset_instance()


def test_explicit_legacy_deepseek_chat_model_is_preserved(monkeypatch, tmp_path):
    config = _build_config(
        monkeypatch,
        tmp_path,
        {"DEEPSEEK_API_KEY": "sk-deepseek-123456", "LITELLM_MODEL": "deepseek/deepseek-chat"},
    )
    try:
        assert config.litellm_model == "deepseek/deepseek-chat"
    finally:
        Config.reset_instance()


# ---------------------------------------------------------------------------
# 8. config_registry
# ---------------------------------------------------------------------------

def test_registry_registers_deepseek_search_fields_as_data_source():
    from src.core.config_registry import build_schema_response, get_field_definition

    field = get_field_definition("DEEPSEEK_SEARCH_API_KEYS")
    assert field["category"] == "data_source"
    assert field["ui_control"] == "password"
    assert field["is_sensitive"] is True
    assert field["validation"]["multi_value"] is True

    assert get_field_definition("DEEPSEEK_SEARCH_ENABLED")["data_type"] == "boolean"
    assert get_field_definition("DEEPSEEK_SEARCH_NEED_CONTENT")["category"] == "data_source"
    assert get_field_definition("DEEPSEEK_SEARCH_BASE_URL")["default_value"] == "https://api.deepseek.com"

    schema = build_schema_response()
    data_source = next(c for c in schema["categories"] if c["category"] == "data_source")
    keys = {f["key"] for f in data_source["fields"]}
    assert {
        "DEEPSEEK_SEARCH_API_KEYS",
        "DEEPSEEK_SEARCH_ENABLED",
        "DEEPSEEK_SEARCH_BASE_URL",
        "DEEPSEEK_SEARCH_NEED_CONTENT",
    } <= keys
