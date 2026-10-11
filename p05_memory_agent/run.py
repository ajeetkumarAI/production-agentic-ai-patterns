"""Demo: python -m p05_memory_agent.run

Session 1 on "laptop" -> sync -> session 2 on "phone" recalls what was learned.
"""
from __future__ import annotations

from common import config
from common.llm import get_llm

from .agent import MemoryAgent
from .memory import MemoryStore, sync


def main() -> None:
    llm = get_llm()
    laptop = MemoryStore(config.DATA_DIR / "p05_laptop.db")
    phone = MemoryStore(config.DATA_DIR / "p05_phone.db")
    user = "ajeet"

    print("== Session 1 (laptop) ==")
    a = MemoryAgent(llm, laptop, user, buffer_tokens=400)
    for msg in ["Hi! I'm working on a fraud detection thesis using XGBoost and SHAP.",
                "I prefer concise answers with Python code examples.",
                "My deadline for the thesis draft is end of November.",
                "Can you suggest how to handle class imbalance?"]:
        print(f"USER: {msg}\nBOT : {a.chat(msg)[:200]}\n")
    print("compressions:", a.buffer.compressions, "| running summary:", a.buffer.summary[:200])
    print("session summary:", a.end_session()[:200])

    print("\n== Sync laptop <-> phone ==", sync(laptop, phone))

    print("\n== Session 2 (phone) ==")
    b = MemoryAgent(llm, phone, user)
    q = "What model was I using for my thesis, and how should you format answers for me?"
    print(f"USER: {q}\nBOT : {b.chat(q)[:300]}")
    print("recalled:", [(m.text, round(m.score, 3)) for m in b.last_recalled])
    print("previous-session context:", b.previous_session[:200])


if __name__ == "__main__":
    main()
