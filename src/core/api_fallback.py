"""
API Fallback Module - Tự động chuyển model khi gặp lỗi 503.

Sử dụng:
    from core.api_fallback import call_with_fallback, create_agent_with_fallback
"""
from __future__ import annotations

import asyncio
from typing import Callable, Any, TypeVar

from core.config import (
    get_red_provider,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    get_next_fallback_model,
    get_gemini_fallback_models,
)

T = TypeVar('T')


class APIFallbackError(Exception):
    """Exception khi tất cả các model đều thất bại."""
    def __init__(self, message: str, last_error: Exception = None):
        super().__init__(message)
        self.last_error = last_error


def is_retryable_error(e: Exception) -> bool:
    """Kiểm tra xem lỗi có nên retry với model khác không.

    Các lỗi nên retry:
    - 503 Service Unavailable
    - 429 Rate Limit
    - 500/502/503/504 Server errors
    - Connection errors
    - SSL errors (có thể tạm thời)
    """
    error_str = str(e).lower()
    error_code = getattr(e, 'code', None)

    # Check error codes
    if error_code in (429, 500, 502, 503, 504):
        return True

    # Check error messages
    retryable_patterns = [
        '503', 'rate limit', 'rate_limit',
        'unavailable', 'overloaded', 'quota',
        'connection', 'timeout', 'ssl',
        'certificate verify failed',
    ]

    for pattern in retryable_patterns:
        if pattern in error_str:
            return True

    return False


async def call_with_fallback(
    func: Callable[..., Any],
    *args,
    current_model: str = None,
    max_retries: int = None,
    **kwargs
) -> Any:
    """Gọi function với fallback mechanism.

    Args:
        func: Async function cần gọi
        *args: Arguments cho function
        current_model: Model hiện tại
        max_retries: Số lần retry tối đa (mặc định = số fallback models)
        **kwargs: Keyword arguments cho function

    Returns:
        Kết quả từ function

    Raises:
        APIFallbackError: Khi tất cả models đều thất bại
    """
    if get_red_provider() != PROVIDER_GEMINI:
        # Không có fallback cho OpenAI
        return await func(*args, **kwargs)

    models = get_gemini_fallback_models()
    if max_retries is None:
        max_retries = len(models)

    last_error = None

    for attempt in range(max_retries):
        try:
            result = await func(*args, **kwargs)
            if current_model and attempt > 0:
                print(f"[INFO] Successfully switched to model: {current_model}")
            return result

        except Exception as e:
            last_error = e

            if not is_retryable_error(e):
                # Không phải lỗi có thể retry
                raise

            # Thử model tiếp theo
            next_model = get_next_fallback_model(current_model)

            if next_model is None:
                print(f"[WARN] No more fallback models available")
                raise APIFallbackError(
                    f"All models exhausted. Last error: {e}", last_error=e
                )

            current_model = next_model
            print(f"[WARN] Model {kwargs.get('model', current_model)} failed: {type(e).__name__}")
            print(f"[INFO] Switching to fallback model: {next_model}")

            # Cập nhật model trong kwargs nếu có
            if 'model' in kwargs:
                kwargs['model'] = next_model

    raise APIFallbackError(
        f"Max retries ({max_retries}) exceeded", last_error=last_error
    )


def create_agent_with_fallback(
    create_func: Callable[..., tuple],
    agent_name: str = "agent",
    **kwargs
) -> tuple:
    """Tạo agent với fallback model.

    Args:
        create_func: Hàm tạo agent (ví dụ: create_red_agent_default)
        agent_name: Tên agent để hiển thị
        **kwargs: Arguments cho hàm tạo

    Returns:
        Tuple (agent, runner)
    """
    if get_red_provider() != PROVIDER_GEMINI:
        return create_func(**kwargs)

    models = get_gemini_fallback_models()
    last_error = None

    for model in models:
        try:
            print(f"[INFO] Trying to create {agent_name} with model: {model}")
            result = create_func(model=model, **kwargs)
            print(f"[SUCCESS] {agent_name} created with model: {model}")
            return result
        except Exception as e:
            last_error = e
            print(f"[WARN] Failed to create {agent_name} with {model}: {type(e).__name__}")

            if not is_retryable_error(e):
                break

    raise APIFallbackError(
        f"Failed to create {agent_name} with any model", last_error=last_error
    )


# Decorator cho easy use
def with_fallback(current_model: str = None):
    """Decorator để tự động retry với fallback.

    Usage:
        @with_fallback()
        async def my_api_call(model=None):
            return await some_llm_call(model=model)
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            return await call_with_fallback(
                func, *args,
                current_model=current_model,
                **kwargs
            )
        return wrapper
    return decorator
