import json
import os
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage


llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENAI_API_KEY"]
)



class InterviewState(TypedDict):
    domain: str
    history: List[Dict[str, str]]
    topics: List[str]
    difficulty: str
    last_question: str
    last_answer: str
    observer: Dict[str, Any]
    factcheck: str
    decision: str
    rounds: int
    finished: bool
    final_report: Dict[str, Any]
    participant_name: str
    log_turn_id: int
    current_agent_question: str
    first_question_sent: bool
    question_asked_in_current_cycle: bool
    next_question_override: str
    thought_buffer: List[str]
    log_file: str
    scenario_id: str

 




ASK_PROMPT = """
Ты технический интервьюер.
Ты проводишь интервью строго в этом домене.
Задавай следующий вопрос кандидату, строго логически связанный с предыдущими ответами (не из случайного списка).
Нельзя повторять первый вопрос никогда. Общайся вежливо.

Условия:
- вопросы должны строиться на том, что кандидат отвечал в последних трех обменах
- не повторяй вопросы
- адаптируй сложность: {difficulty}
- коротко
История:
"""


OBSERVER_PROMPT = """
Ты Observer агент интервью.
Проанализируй ответ кандидата. Проверь, относится ли следующий ответ кандидата к вопросу.
Если нет — верни score=0 и instruction вернуть разговор к теме.
Если да — оцени нормально.

Вопрос: {state['last_question']}

Верни JSON:
{
 "score": 0-100,
 "strengths": [],
 "gaps": [],
 "instruction": "что должен сделать интервьюер дальше"
}

Ответ:
"""

FACTCHECK_PROMPT = """
Проверь ТОЛЬКО объективные технические утверждения.
Если в тексте нет проверяемых технических фактов — ответь OK.
Игнорируй фразы про незнание, сомнение, эмоции, опыт.
Примеры, которые НУЖНО игнорировать:
- "я не знаю"
- "я не понимаю"
- "мне кажется"

Если есть фактическая ошибка — напиши:
HALLUCINATION: <почему>

Иначе:
OK

Ответ:
"""




DECISION_PROMPT = """
Ты Decision Agent интервью.

Выбери ОДНО действие:
ask_harder
ask_simpler
clarify
challenge
teach
next_topic
finish

Верни только одно слово, без пояснений.

Observer:
{observer}

FactCheck:
{fact}

Ответ:
"""

TEACHER_PROMPT = """
Ты технический интервьюер.

Кандидат не понял или не знает термин.
Твоя задача — объяснить его буквально одним предложением, просто и понятно по-человечески, важно не давать готового решения задачи.

Тема:
{topic}

Вопрос:
{question}

Ответ кандидата:
{answer}

Объясни.
"""


FINAL_PROMPT = """
Ты Senior Hiring Manager и технический экзаменатор.

Проанализируй интервью строго по логам turns.
Кандидат может уклоняться, не отвечать или задавать встречные вопросы — это тоже сигнал.

Твоя задача:
1. Выставить уровень кандидата (grade) по фактическим проявленным знаниям.
2. Сформировать verdict (hire).
3. Найти:
   - что кандидат реально показал
   - чего не показал
   - какие темы были затронуты, но не раскрыты
4. Проанализируй Soft Skills кандидата по логам интервью.

Оцени:
- Clarity — насколько ясно и связно кандидат формулирует мысли.
- Honesty — признает ли незнание или пытается выкрутиться и уходить от ответа.
- Engagement — проявляет ли вовлеченность (задает вопросы, отвечает на вопросы эмоционально, подробно, по теме интервью, а не игнорирует его). Важно:
Если последний ответ кандидата — "stop", "стоп", "exit", "quit" или аналогичная команда завершения,
считай это СИСТЕМНЫМ действием пользователя, а не проявлением его мотивации, вовлеченности или soft skills!!
Не используй факт остановки интервью при оценке Clarity, Honesty или Engagement.
Оцени soft skills только по содержательным ответам кандидата.


Для каждого пункта дай короткую оценку (начинающий, опытный, эксперт) и 1 предложение объяснения.

Важно:
Если кандидат не дал ни одного технического ответа — это считается Knowledge Gap.

Для каждого выявленного gap:
- укажи, КАКОЙ правильный технический ответ ожидался (кратко, по делу)

В Resources указывай только валидные ссылки

Верни СТРОГО такой JSON:

{
 "Grade": "Junior | Middle | Senior | 0",
 "Hire": "Hire | No hire| Strong hire",
 "Confidence": 0-100,
 "Strengths": [
    "конкретные проявленные навыки"
 ],
 "gaps": [
    {
      "Topic": "тема",
      "Knowledge Gaps": "что кандидат не смог объяснить",
      "Correct Answer": "как должен был звучать правильный технический ответ"
    }
 ],
  "soft_skills": {
    "clarity": "оценка, так как объяснение",
    "honesty": "оценка, так как объяснение",
    "engagement": "оценка, так как объяснение"
 },
 "Next Steps": [
    "конкретные темы или технологии для изучения"
 ],
 "Resources": [
         "ссылка на официальную документацию или статью",
         "еще одна полезная ссылка"
      ]
}

Логи интервью:
"""





def call_llm(prompt: str) -> str:
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content

def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("no json found")
    return json.loads(text[start:end])



LOG_FILE = "interview_log.json"



def init_log(participant_name: str, scenario_id: str = "001"):
    """Создание стартового файла лога интервью."""
    filename = f"interview_log_{scenario_id}.json"
    data = {
        "participant_name": participant_name,
        "turns": [],
        "final_feedback": ""
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[LOG INIT] → {os.path.abspath(filename)}")
    return filename



def append_turn(log_file: str, turn_id: int, agent_msg: str, user_msg: str, thoughts_list: List[str]):
    """Сохраняет ход интервью в JSON, без вывода в консоль."""
    if not os.path.exists(log_file):
        data = {"participant_name": "", "turns": [], "final_feedback": ""}
    else:
        with open(log_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    thoughts = ""
    for t in thoughts_list:
        thoughts += t.rstrip() + "\n"

    turn = {
        "turn_id": turn_id,
        "agent_visible_message": agent_msg,
        "user_message": user_msg,
        "internal_thoughts": [t.strip() for t in thoughts_list] 
    }

    data["turns"].append(turn)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def write_final_feedback(log_file: str, feedback: dict):
    """Сохраняет финальный отчет в JSON и выводит только финальный результат в консоль."""
    if not os.path.exists(log_file):
        data = {"participant_name": "", "turns": [], "final_feedback": feedback}
    else:
        with open(log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["final_feedback"] = feedback

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


    print("\n===== FINAL REPORT =====")
    print(json.dumps(feedback, indent=2, ensure_ascii=False))






def ask_node(state: InterviewState):
    FIRST_Q = f"Привет! Расскажи про свой опыт."


    if state.get("next_question_override"):
        q = state.pop("next_question_override", "").strip()
        if not q.endswith("?"):
            q += "?"
        state["last_question"] = q
        state["current_agent_question"] = q
        state["history"].append({"role": "interviewer", "text": q})
        state["rounds"] += 1
        print("\nInterviewer:", q)
        return state

    if state.get("question_asked_in_current_cycle", False):
        state["question_asked_in_current_cycle"] = False
        return state

    previous_questions = [h["text"] for h in state.get("history", []) if h["role"] == "interviewer"]

    if not state.get("first_question_sent", False) and FIRST_Q not in previous_questions:
        q = FIRST_Q
        state["first_question_sent"] = True
    else:
        prompt = ASK_PROMPT.format(
    difficulty=state["difficulty"])

        context_window = state["history"][-6:]
        prompt += json.dumps(context_window, ensure_ascii=False)
        q = call_llm(prompt).strip()
        if not q:
            last_candidate_answer = state["history"][-1]["text"] if state["history"] else ""
            q = f"Можешь подробнее рассказать про: {last_candidate_answer}"
        if q in previous_questions:
            last_candidate_answer = state["history"][-1]["text"] if state["history"] else ""
            q = f"Можешь привести пример или уточнить: {last_candidate_answer}"

    state["last_question"] = q
    state["current_agent_question"] = q
    state["history"].append({"role": "interviewer", "text": q})
    state["rounds"] += 1
    state["question_asked_in_current_cycle"] = True

    print("\nInterviewer:", q)
    return state










def teach_node(state: InterviewState):
    prompt = TEACHER_PROMPT.format(
        topic=state["topics"][-1] if state["topics"] else "",
        question=state["last_question"],
        answer=state["last_answer"]
    )

    msg = call_llm(prompt)

    print("\nInterviewer (explain):", msg)
    state["history"].append({"role": "interviewer", "text": msg})

    return state



def candidate_node(state: InterviewState):
    a = input("Candidate: ").strip()
    stop_words = ["стоп игра", "стоп", "stop", "выход", "exit", "quit"]

    state["last_answer"] = a
    state["history"].append({"role": "candidate", "text": a})

    if a.lower() in stop_words:
        state["finished"] = True

        append_turn(
        log_file=state["log_file"],
        turn_id=state.get("log_turn_id", 1),
        agent_msg=state.get("current_agent_question", ""),
        user_msg=a,
        thoughts_list=["[System]: interview stopped by candidate"])

        state["log_turn_id"] += 1
        print("\n[System]: Interview stopped by candidate")
        return state

    return state




def observer_node(state: InterviewState):
    prompt = OBSERVER_PROMPT.replace("{state['last_question']}", state["last_question"])
    resp = call_llm(prompt + "\n\nОтвет кандидата:\n" + state["last_answer"])
    try:
        data = json.loads(resp)
    except:
        data = {"score": 50, "strengths": [], "gaps": [], "instruction": "clarify"}

    print("\n[Hidden Reflection]")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    state["thought_buffer"].append(f"[Observer]: {json.dumps(data, ensure_ascii=False)}")
    state["observer"] = data
    return state




def factcheck_node(state: InterviewState):
    fc = call_llm(FACTCHECK_PROMPT + state["last_answer"])
    print("\n[FactCheck]:", fc)
    state["factcheck"] = fc
    state["thought_buffer"].append(f"[FactCheck]: {fc}")
    return state



def decision_node(state: InterviewState):
    obs = state.get("observer", {})
    gaps = obs.get("gaps", [])
    answer = state.get("last_answer", "").lower()
    decision = None

    candidate_questions = [
        "какие у вас", "какие условия", "условия труда", "что вы предлагаете",
        "что за компания", "что за проект", "что за команда", "чем занимается",
        "испытательный срок", "какие задачи", "что буду делать", "что входит"
    ]
    if "?" in answer or any(x in answer for x in candidate_questions):
        decision = "answer_candidate_question"
    elif any(x in answer for x in ["что такое", "что значит", "не понимаю", "объясни"]):
        decision = "teach"
    if not decision:
        for g in gaps:
            if any(x in g.lower() for x in ["поясни", "что такое", "не понимает"]):
                decision = "teach"
                break
    if not decision and state.get("factcheck", "").startswith("HALLUCINATION"):
        decision = "challenge"
    if not decision:
        prompt = DECISION_PROMPT.format(
            observer=json.dumps(state["observer"], ensure_ascii=False),
            fact=state["factcheck"]
        )
        decision = call_llm(prompt).strip().lower()
    
    
    if state["last_answer"].lower() in ["нет не хочу я", "не хочу", "не буду отвечать"]:
        decision = "next_topic"
    state["decision"] = decision
    if decision == "finish" or state["rounds"] >= 12:
        state["finished"] = True

    state["thought_buffer"].append(f"[Decision]: {decision}")
    append_turn(
        state["log_file"],
        state["log_turn_id"],
        state.get("current_agent_question", ""),
        state["last_answer"],
        state["thought_buffer"]
    )
    state["thought_buffer"] = []  
    state["log_turn_id"] += 1




    return state





def answer_candidate_question_node(state: InterviewState):
    prompt = f"""Ты технический интервьюер.
Отвечай кандидату кратко и по делу одним предложением, на его вопрос, без перехода к следующему техническому вопросу.

Вопрос кандидата: {state['last_answer']}
Отвечай дружелюбно, понятно и кратко, одним предложением.
"""
    reply = call_llm(prompt)
    print("\nInterviewer (ответ на вопрос):", reply)
    state["history"].append({"role": "interviewer", "text": reply})
    state["question_asked_in_current_cycle"] = False
    state["next_question_override"] = None

    return state



def action_node(state: InterviewState):
    last_q = state.get("last_question", "")
    last_a = state.get("last_answer", "")
    instr = state.get("observer", {}).get("instruction", "").strip()

    # Если нужно просто задать следующий технический вопрос
    prompt = f"Ты технический интервьюер, общайся как человек.\n"
    prompt += f"- Предыдущий вопрос: {last_q}\n"
    prompt += f"- Последний ответ кандидата: {last_a}\n"
    if instr:
        prompt += f"- Совет для следующего вопроса: {instr}\n"
    prompt += "- Сформулируй короткий, ясный вопрос, связанный с предыдущими ответами.\n"

    q = call_llm(prompt).strip()
    if not q:
        q = "Можешь подробнее рассказать об этом?"
    if not q.endswith("?"):
        q += "?"
    state["next_question_override"] = q
    return state






def topic_tracker_node(state: InterviewState):
    topic = call_llm(
    f"Ты технический интервьюер."
    f"Назови кратко тему вопроса: {state['last_question']}"
)

    state["topics"].append(topic)
    return state





def final_node(state: InterviewState):
    """Формирование финального отчета на основе реальных ходов кандидата."""
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    rep = call_llm(FINAL_PROMPT + json.dumps(log_data["turns"], ensure_ascii=False))
    try:
        final_feedback = extract_json(rep)
    except Exception as e:
        print("\n[Final parse error]:", e)
        print("\n[Raw LLM output]:\n", rep)
        final_feedback = {
            "grade": "unknown",
            "hire": "unknown",
            "confidence": 0,
            "strengths": [],
            "gaps": ["final parse failed"],
            "next_steps": []
        }

    write_final_feedback(state["log_file"], final_feedback)


    print("\n===== FINAL REPORT =====")
    print(json.dumps({
        "participant_name": state["participant_name"],
        "turns": log_data["turns"],
        "final_feedback": final_feedback
    }, indent=2, ensure_ascii=False))

    state["final_report"] = final_feedback
    return state



builder = StateGraph(InterviewState)

builder.add_node("ask", ask_node)
builder.add_node("candidate", candidate_node)
builder.add_node("observe", observer_node)
builder.add_node("fact", factcheck_node)
builder.add_node("decide", decision_node)
builder.add_node("act", action_node)
builder.add_node("topic", topic_tracker_node)
builder.add_node("final", final_node)
builder.add_node("teach", teach_node)


builder.set_entry_point("ask")

builder.add_edge("ask", "candidate")
builder.add_conditional_edges(
    "candidate",
    lambda s: "final" if s["finished"] else "observe"
)


builder.add_edge("observe", "fact")
builder.add_edge("fact", "decide")
builder.add_edge("act", "topic")


def route(state):
    if state["finished"]:
        return "final"

    if state.get("decision") == "next_topic":
        return "ask"


    last_answer = state.get("last_answer", "").lower()
    refusal_phrases = ["нет не хочу я", "не хочу", "не буду отвечать"]
    if last_answer in refusal_phrases:

        state["decision"] = "next_topic"
        return "ask"  


    return "ask"




builder.add_conditional_edges("topic", route)
builder.add_edge("final", END)
builder.add_conditional_edges(
    "decide",
    lambda s: s["decision"],
    {
        "teach": "teach",
        "answer_candidate_question": "answer_candidate_question",   
        "ask_harder": "act",
        "ask_simpler": "act",
        "clarify": "act",
        "challenge": "act",
        "next_topic": "act",
        "finish": "final"
    }
)


builder.add_edge("teach", "ask")
builder.add_node("answer_candidate_question", answer_candidate_question_node)
builder.add_edge("answer_candidate_question", "ask")





graph = builder.compile()


name = input("Введите ФИО кандидата: ")

scenario_id = "001"
log_file = init_log(name, scenario_id)

init_state: InterviewState = {
    "history": [],
    "topics": [],
    "difficulty": "medium",
    "last_question": "",
    "last_answer": "",
    "observer": {},
    "factcheck": "",
    "decision": "",
    "rounds": 0,
    "finished": False,
    "final_report": {},
    "first_question_sent": False,
    "question_asked_in_current_cycle": False,
    "participant_name": name,
    "log_turn_id": 1,
    "current_agent_question": "",
    "next_question_override": "",
    "thought_buffer": [],
    "log_file": log_file,
    "scenario_id": scenario_id
}



graph.invoke(init_state)
