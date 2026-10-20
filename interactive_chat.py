"""
Interactive Terminal Chat for Legal QA Backend.
Run this script to chat directly with the legal assistant in actionable mode:
    python interactive_chat.py
"""

import sys
import httpx

API_BASE = "http://127.0.0.1:8080"


def main():
    print("=" * 60)
    print("      INDIAN LEGAL QA - INTERACTIVE CHAMBERS CHAT      ")
    print("                 Mode: ACTIONABLE                      ")
    print("=" * 60)
    print("Type your legal problem in plain language. Type 'exit' to quit.\n")

    client = httpx.Client(base_url=API_BASE, timeout=120.0)

    # 1. Create a new conversation
    try:
        resp = client.post("/api/conversations", json={"mode": "actionable"})
        resp.raise_for_status()
        conv_data = resp.json()
        conv_id = conv_data["conversation_id"]
        print(f"[Connected to session: {conv_id[:10]}...]\n")
    except Exception as exc:
        print(f"[ERROR] Could not connect to backend at {API_BASE}: {exc}")
        print("Please ensure uvicorn is running: uvicorn app:app --port 8080")
        return

    # 2. Conversation loop
    has_given_initial_advice = False

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting session. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Exiting session. Goodbye!")
            break

        print("\nAdvocate: Thinking...", end="\r", flush=True)

        try:
            msg_resp = client.post(
                f"/api/conversations/{conv_id}/messages",
                json={"message": user_input, "content": user_input},
            )
            msg_resp.raise_for_status()
            data = msg_resp.json()
        except Exception as exc:
            print(f"\n[ERROR] Request failed: {exc}")
            continue

        resp_type = data.get("type")

        if resp_type == "follow_up":
            msg = data.get("message") or data.get("answer", "")
            print(f"\nAdvocate:\n{msg}\n")
        elif resp_type == "final_answer":
            answer = data.get("answer") or data.get("message", "")
            if not has_given_initial_advice:
                print("\n" + "=" * 60)
                print("              LEGAL ADVICE & ACTION PLAN               ")
                print("=" * 60)
                print(answer)
                print("\n" + "=" * 60)
                print("\n[Consultation active — You can ask follow-up questions, discuss specific points, explore 'what if' scenarios, or ask about another legal matter. Type 'exit' to quit.]")
                has_given_initial_advice = True
            else:
                print(f"\nAdvocate:\n{answer}\n")
        else:
            print(f"\nResponse: {data}")


if __name__ == "__main__":
    main()
