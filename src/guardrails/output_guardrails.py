"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

def content_filter(response: str) -> dict:
    """
    Quét và che giấu (redact) PII và secrets trong câu trả lời từ LLM.
    Trả về dict gồm:
      - safe (bool): True nếu không có vấn đề nào cần redact, False nếu có vi phạm
      - issues (list[str]): danh sách mô tả các lỗi phát hiện
      - redacted (str): chuỗi đã được thay thế dữ liệu nhạy cảm bằng [REDACTED]
    """
    if not response:
        return {"safe": True, "issues": [], "redacted": ""}

    redacted_text = response
    issues = []

    # Danh sách các mẫu dữ liệu nhạy cảm cần lọc
    sensitive_patterns = [
        # Số điện thoại Việt Nam (+84 hoặc 0x, theo sau là 9 chữ số)
        ("Vietnamese Phone Number", r'(\+84|0)(3|5|7|8|9)[0-9]{8}\b'),
        
        # Email
        ("Email Address", r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'),
        
        # CCCD / CMND (12 chữ số)
        ("Citizen ID (CCCD)", r'\b0[0-9]{11}\b'),
        
        # API Key (định dạng sk-...)
        ("API Key", r'sk-[a-zA-Z0-9_\-]{10,}'),
        
        # Password / Secret trực tiếp (admin123 hoặc password: ...)
        ("Admin Password / Secret", r'(?i)(admin_password|password|mật khẩu)[\s:=]+[\w\-@!#$%^&*]+|admin123')
    ]

    for label, pattern in sensitive_patterns:
        matches = re.findall(pattern, redacted_text)
        if matches:
            issues.append(f"Detected {label}")
            redacted_text = re.sub(pattern, '[REDACTED]', redacted_text)

    is_safe = (len(issues) == 0)

    return {
        "safe": is_safe,
        "issues": issues,
        "redacted": redacted_text
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        # 1. Gọi content_filter(response_text)
        filter_result = content_filter(response_text)
        if not filter_result["safe"] or len(filter_result["issues"]) > 0:
            self.redacted_count += 1
            response_text = filter_result["redacted"]
            # Cập nhật nội dung llm_response bằng bản đã che chắn [REDACTED]
            llm_response.content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=response_text)],
            )

        # 2. Nếu bật LLM Judge: gọi llm_safety_check(response_text)
        if self.use_llm_judge and "llm_safety_check" in globals():
            import inspect
            is_unsafe = False
            try:
                # Hỗ trợ cả hàm sync và async cho llm_safety_check
                if inspect.iscoroutinefunction(llm_safety_check):
                    judge_result = await llm_safety_check(response_text)
                else:
                    judge_result = llm_safety_check(response_text)

                # Phân tích kết quả trả về từ judge
                if isinstance(judge_result, bool):
                    is_unsafe = not judge_result
                elif isinstance(judge_result, str):
                    is_unsafe = judge_result.strip().upper() in ["UNSAFE", "BLOCK"]
                elif isinstance(judge_result, dict):
                    is_unsafe = not judge_result.get("safe", True)
            except Exception:
                is_unsafe = False

            if is_unsafe:
                self.blocked_count += 1
                safe_message = "Yêu cầu bị từ chối: Phản hồi vi phạm chính sách an toàn nội dung của VinBank."
                llm_response.content = types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=safe_message)],
                )

        # 3. Trả về llm_response đã được xử lý an toàn
        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
