from __future__ import annotations

from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


def build_agentic_messages(
    step: int,
    tool_tokens: int = 256,
    request_id: str | None = None,
    final_instruction: str | None = None,
) -> list[Message]:
    if step < 1:
        raise ValueError("step must be at least 1")
    if tool_tokens < 0:
        raise ValueError("tool_tokens cannot be negative")

    system_content = "You are an agent. Use tool results and answer briefly."
    if request_id is not None:
        system_content = f"[REQUEST_ID={request_id}] {system_content}"

    messages: list[Message] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "Plan a 3-day trip. Use the provided tool results."},
    ]
    tool_result_unit = " hotel weather transport"
    for index in range(step - 1):
        result_number = index + 1
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": f"I will use tool result {result_number}.",
                },
                {
                    "role": "user",
                    "content": (
                        f"[TOOL_RESULT_{result_number}]" + tool_result_unit * tool_tokens
                    ),
                },
            ]
        )

    if final_instruction is not None:
        messages.append({"role": "user", "content": final_instruction})
    return messages

