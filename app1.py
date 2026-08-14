import os
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import base64
import pickle
import re
import time
import uuid

import faiss
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from groq import Groq, RateLimitError
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

print("Loading embedding model...")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Loading FAISS index...")
index = faiss.read_index("data/faiss_index.bin")

print("Loading document store...")
with open("data/doc_store.pkl", "rb") as f:
    doc_store = pickle.load(f)

TEXT_MODEL = "openai/gpt-oss-20b"
llm = ChatGroq(model=TEXT_MODEL, temperature=0, api_key=os.environ["GROQ_API_KEY"])

VISION_MODEL = "qwen/qwen3.6-27b"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

PRONOUN_PATTERN = re.compile(
    r"\b(it|its|it's|they|them|their|there|they're|that|this|these|those)\b",
    re.IGNORECASE,
)

THINK_BLOCK_PATTERN = re.compile(
    r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
    re.DOTALL | re.IGNORECASE,
)


def strip_thinking(text):
    return THINK_BLOCK_PATTERN.sub("", text).strip()


MEDICAL_DISCLAIMER = (
    "\n\n⚠️ This is general information only, not medical advice. "
    "Before taking any step, please consult a health specialist or a doctor."
)

GREETING_PATTERN = re.compile(
    r"^\s*(hi+|hello+|hey+|good\s?(morning|afternoon|evening)|"
    r"thanks?( you)?|thank\s?you|bye+|goodbye|see\s?ya|ok(ay)?)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def detect_greeting(query):
    q = query.strip().lower()
    if not GREETING_PATTERN.match(q):
        return None
    if "thank" in q:
        return "You're welcome! Let me know if you have another question."
    if "bye" in q or "goodbye" in q or "see ya" in q:
        return "Take care! Come back anytime you have a health or hospital question."
    return (
        "Hi! I can help with questions about medicines, hospital services, "
        "or general health topics — and you can upload a photo for me to "
        "describe. What would you like to know?"
    )


MEMORY_QUERY_PATTERN = re.compile(
    r"\b(what('?s| is) my name|who am i|do you (remember|know) my name|"
    r"what did i (tell|say)|what('?s| is) my (preference|preferences)|"
    r"remember (what|who) i|do you remember me)\b",
    re.IGNORECASE,
)


def looks_like_memory_query(query):
    return bool(MEMORY_QUERY_PATTERN.search(query))


PERSONAL_INFO_PATTERN = re.compile(
    r"\bmy name is\b|\bcall me\b|\bi am\s+\d{1,3}\s*(years?\s*old)?\b|"
    r"\bi'm\s+\d{1,3}\s*(years?\s*old)?\b|\bmy age is\b",
    re.IGNORECASE,
)


def looks_like_personal_statement(query):
    if "?" in query:
        return False
    if looks_like_advice_query(query):
        return False
    return bool(PERSONAL_INFO_PATTERN.search(query))


def acknowledge_personal_statement(query, memory):
    history_text = format_memory(memory, max_turns=10)
    prompt = f"""The user just told you something about themselves in a
conversation with an AI hospital assistant: "{query}"

Conversation so far:
{history_text}

Reply with a brief, warm one-sentence acknowledgment (use their name if
they gave one, e.g. "Nice to meet you, Swapnil!"), then ask how you can
help. Do not add any medical information, facts, or advice. Do not use
any outside knowledge or database — this is purely a social
acknowledgment.

Output ONLY the reply, nothing else."""
    try:
        return strip_thinking(safe_invoke(prompt).content) or "Nice to meet you! How can I help you today?"
    except Exception:
        return "Nice to meet you! How can I help you today?"


sessions = {}


def get_session_state():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    sid = session["sid"]
    if sid not in sessions:
        sessions[sid] = {
            "last_resolved_query": None,
            "current_topic": None,
            "current_topic_source": None,
            "last_image_context": None,
            "history": [],
            "memory": ConversationBufferMemory(return_messages=True, memory_key="chat_history"),
        }
    return sessions[sid]


def remember_turn(memory, user_text, answer_text):
    clean_answer = answer_text.replace(MEDICAL_DISCLAIMER, "").strip()
    try:
        memory.save_context({"input": user_text}, {"output": clean_answer})
    except Exception:
        pass


def format_memory(memory, max_turns=6, max_chars_per_turn=400):
    try:
        messages = memory.load_memory_variables({}).get("chat_history", [])
    except Exception:
        return "None (this is the beginning of the conversation)."

    lines = []
    for msg in messages[-(max_turns * 2):]:
        role = "User" if msg.type == "human" else "Assistant"
        lines.append(f"{role}: {msg.content[:max_chars_per_turn]}")
    return "\n".join(lines) if lines else "None (this is the beginning of the conversation)."


def answer_from_memory(query, memory):
    history_text = format_memory(memory, max_turns=10)
    if history_text.startswith("None"):
        return None

    prompt = f"""You are continuing an ongoing conversation with a user.

Conversation so far:
{history_text}

New question: "{query}"

Using ONLY the conversation above (no outside knowledge, no medical
database), check whether this question is simply asking you to recall
something the user already told you earlier in this conversation (for
example their name, a preference, or a fact they stated).

If the conversation above contains the answer, answer briefly and
naturally in one or two sentences.
If it does NOT contain the answer, output exactly: NO_MEMORY_MATCH

Answer:"""

    try:
        raw = safe_invoke(prompt).content
    except Exception:
        return None

    result = strip_thinking(raw).strip()
    if result == "NO_MEMORY_MATCH" or not result:
        return None
    return result


def safe_invoke(prompt, retries=1):
    for attempt in range(retries + 1):
        try:
            return llm.invoke(prompt)
        except RateLimitError as e:
            if attempt == retries:
                raise
            m = re.search(r"try again in ([\d.]+)([ms])", str(e))
            wait = min(float(m.group(1)) * (60 if m.group(2) == "m" else 1), 30) if m else 10
            time.sleep(wait)


def safe_vision_call(messages, retries=1):
    for attempt in range(retries + 1):
        try:
            return groq_client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
                reasoning_format="hidden",
            )
        except RateLimitError as e:
            if attempt == retries:
                raise
            m = re.search(r"try again in ([\d.]+)([ms])", str(e))
            wait = min(float(m.group(1)) * (60 if m.group(2) == "m" else 1), 30) if m else 10
            time.sleep(wait)


def extract_subject(vision_answer):
    try:
        prompt = f"""Description: "{vision_answer[:400]}"

In 3-8 words, name the specific subject of this description as a noun
phrase (e.g. "the Patient Information Form", "Calpol 500 paracetamol
tablets"). Output ONLY the noun phrase, nothing else."""
        subject = safe_invoke(prompt).content.strip()
        return subject or None
    except Exception:
        return None


def extract_topic_noun(resolved_query):
    try:
        prompt = f"""Question: "{resolved_query}"

In 1-4 words, name the core medical subject/topic of this question
(e.g. "diabetes", "cholera", "Calpol paracetamol"). Output ONLY the
noun phrase, nothing else."""
        topic = safe_invoke(prompt).content.strip()
        return topic or resolved_query
    except Exception:
        return resolved_query


def correct_spelling(query):
    try:
        prompt = f"""This question was typed into a hospital/medical assistant
chatbot and may contain a misspelled medical term, medicine name, or
health condition. Fix any spelling mistakes, correcting toward the most
likely medical/health term the user meant. Keep the meaning and question
format exactly the same — only fix spelling, don't rephrase, expand, or
answer it.

Question: "{query}"

Output ONLY the corrected question, nothing else."""
        corrected = safe_invoke(prompt).content.strip()
        return corrected or query
    except Exception:
        return query


def rewrite_query(query, history, current_topic=None):
    if not history and not current_topic:
        return query, False

    if not PRONOUN_PATTERN.search(query) and not any(w in query.lower() for w in ["both", "difference", "compare", "two"]):
        return query, False

    user_queries = [msg["user"] for msg in history[-3:] if "user" in msg]
    history_context = "\n".join([f"- {q}" for q in user_queries]) if user_queries else "None"

    topic_line = f'Current topic being discussed: "{current_topic}"' if current_topic else "No established topic yet."

    prompt = f"""{topic_line}

Recent User Questions (for extra context only):
{history_context}

Follow-up Question: "{query}"

Rewrite the Follow-up Question into a single, fully standalone search query by replacing pronouns or comparative references (e.g., "it", "them", "both", "these", "the two of them") with the actual medical topics/diseases/medicines being discussed.

If a "Current topic" is given above, prioritize resolving pronouns to that topic UNLESS the Follow-up Question or Recent User Questions clearly introduce a different subject.

If the user is asking to compare or find differences, ensure ALL referenced subjects are explicitly named in the rewritten query.

If the resulting query contains an abbreviation or acronym (e.g. a disease abbreviation like "TB", an organization name, a drug class) and you are confident what it stands for, also spell out the full term in place of the abbreviation. If you are not confident, leave it as-is.

Output ONLY the rewritten standalone query, nothing else."""

    try:
        rewritten = safe_invoke(prompt).content.strip()
        return (rewritten or query), True
    except Exception:
        if current_topic:
            return PRONOUN_PATTERN.sub(current_topic, query), True
        return query, True


ADVICE_PATTERN = re.compile(
    r"\b(what should i do|what to do|i have\b|i've got|i am having|i'm having|"
    r"how (do|can) i (treat|deal with|fix|cure|manage)|"
    r"any (remedy|remedies|solution|solutions|cure)|help( me)? with|"
    r"suffering from|down with)\b",
    re.IGNORECASE,
)


def looks_like_advice_query(query):
    return bool(ADVICE_PATTERN.search(query))


def normalize_advice_query(query):
    try:
        prompt = f"""This question was typed into a hospital/medical assistant
chatbot in conversational, first-person form (e.g. "I have X, what should
I do"). Rewrite it as a short, neutral, document-style query naming the
medical condition/symptom and what's wanted (e.g. "treatment and remedies
for acidity"). Do not answer the question, do not add medical advice, and
do not invent a condition that isn't named — just restate the topic being
asked about.

Question: "{query}"

Output ONLY the rewritten query, nothing else."""
        rewritten = safe_invoke(prompt).content.strip()
        return rewritten or query
    except Exception:
        return query


def choose_k(query, default_k=6, broad_k=10):
    q = query.lower()
    triggers = [
        "compare", "difference", "vs", "versus", "list", "all",
        "which", "best", "summary", "what is", "what are", "explain",
        "treatment", "remedy", "remedies", "cure", "manage", "relief",
    ]
    return broad_k if any(t in q for t in triggers) else default_k


def vector_search(search_query, k=6):
    query_embedding = embedder.encode([search_query], convert_to_numpy=True)
    _, indices = index.search(query_embedding.astype("float32"), k=k)
    context, sources, doc_types = "", set(), set()
    for i, idx in enumerate(indices[0], start=1):
        doc = doc_store[idx]
        sources.add(doc.metadata.get("source", "Unknown"))
        doc_types.add(doc.metadata.get("type", "unknown"))
        context += f"\nDocument {i}:\n{doc.page_content[:2500]}\n"
    return context, sources, doc_types


def get_topic_instructions(doc_types):
    if "medicine" in doc_types:
        return (
            "- List side effects per product, naming each product.\n"
            "- If a substance appears only inside other medicines' Composition "
            "field (no standalone definition document), that IS information "
            "about it: describe it as the active ingredient shared by those "
            "medicines, and combine their Uses/Side Effects into one answer. "
            "Do not say the substance 'wasn't found' if it clearly appears as "
            "an ingredient — go straight to answering from that evidence."
        )
    if "hospital_info" in doc_types:
        return "- List relevant hospitals with location, rating, and services."
    return (
        "- Combine information from ALL retrieved documents relevant to the "
        "question, not just the single best-matching one — different "
        "documents may each contribute a different piece (definition, "
        "causes, symptoms, when to see a doctor, treatment)."
    )


NOT_FOUND_MESSAGE = "I couldn't find that information in the knowledge base."


def _generate_answer(search_query, k, memory, image_context=None):
    context, sources, doc_types = vector_search(search_query, k=k)
    topic_instructions = get_topic_instructions(doc_types)

    history_text = format_memory(memory)

    image_context_block = ""
    if image_context:
        image_context_block = f"""
Context From a Previously Uploaded Image (the user is likely asking a
follow-up about this):
{image_context}

If the Retrieved Documents below don't cover this specific product but
the Image Context identifies it clearly, you may answer using well
established general medical knowledge about the identified drug/condition,
clearly noting it comes from general knowledge rather than the knowledge
base, and skip the "{NOT_FOUND_MESSAGE}" fallback in that case.
"""

    prompt = f"""You are an AI Hospital Assistant.
Use the Retrieved Documents below for facts, plus the Image Context if given.
{image_context_block}
{topic_instructions}
- Give a complete answer: if several retrieved documents each add a
  different piece of the picture, merge them into one coherent answer
  rather than repeating only the shortest or first relevant sentence.
- Use the Recent Conversation Context to understand the flow of the chat,
  but ALWAYS base factual medical answers on the Retrieved Documents.
- Never say "I couldn't find that information" and then answer anyway —
  if the retrieved documents give you enough to answer, just answer.
- Don't invent facts.
- Don't mention information not asked about (e.g. no side effects unless asked).
- Before answering, check that the retrieved documents are actually ABOUT
  the specific thing being asked, not just textually similar to the
  question's wording. Sharing a word or word-fragment with the question
  is not relevance — e.g. a medicine product literally named "Fullform"
  is NOT a relevant match for a question asking what an abbreviation
  stands for. If the documents don't genuinely address the question's
  actual subject, treat it as no relevant match and say so (see below).
- This information is educational, not a substitute for professional medical
  advice — do not tell the user what dose to take, what to start/stop, or
  otherwise phrase the answer as an instruction to act on unsupervised.
- Structure the answer for readability: start with a one-sentence summary,
  then use short markdown headers (## Header) to separate distinct parts
  of a multi-part answer (e.g. Causes, Symptoms, Treatment, Side Effects),
  and "- " bullet points for lists within each part. For a simple
  single-fact answer, plain sentences are fine — don't force structure
  where there's only one point to make.
- Be reasonably detailed: include the specific facts from the retrieved
  documents (names, numbers, conditions), not just a vague summary — but
  stay focused on what was asked, don't pad with restated information.
- If NONE of the retrieved documents are relevant, say exactly:
  "{NOT_FOUND_MESSAGE}"

Recent Conversation Context:
{history_text}

Retrieved Documents:
{context}

Question: {search_query}
Answer:"""

    response = safe_invoke(prompt)
    answer = strip_thinking(response.content)
    return answer, sources


ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,6}\b|\b(?:[A-Za-z]\.){2,}[A-Za-z]?\b")


def looks_like_acronym(query):
    return bool(ACRONYM_PATTERN.search(query))


def expand_acronyms(query):
    try:
        prompt = f"""This question was typed into a hospital/medical
assistant chatbot and may contain an abbreviation or acronym (e.g. an
organization name like WHO, a medical term, a drug class). If you are
confident what it stands for, rewrite the question spelling out the full
term instead of the abbreviation, keeping everything else exactly the
same. If you are not confident, or there is no abbreviation, return the
question completely unchanged.

Question: "{query}"

Output ONLY the resulting question, nothing else."""
        expanded = safe_invoke(prompt).content.strip()
        return expanded or query
    except Exception:
        return query


def _append_history(state, kind, user_text, question, answer):
    state["history"].append({
        "id": str(uuid.uuid4()),
        "ts": datetime.utcnow().isoformat(),
        "kind": kind,
        "user": user_text,
        "assistant": answer,
        "question": question,
        "answer": answer,
    })
    state["history"] = state["history"][-10:]


def ask(query, state, k=None):
    memory = state["memory"]

    greeting_reply = detect_greeting(query)
    if greeting_reply:
        remember_turn(memory, query, greeting_reply)
        _append_history(state, "text", query, query, greeting_reply)
        return greeting_reply, []

    if looks_like_personal_statement(query):
        ack = acknowledge_personal_statement(query, memory)
        remember_turn(memory, query, ack)
        _append_history(state, "text", query, query, ack)
        return ack, []

    if looks_like_memory_query(query):
        memory_answer = answer_from_memory(query, memory)
        if memory_answer is not None:
            remember_turn(memory, query, memory_answer)
            _append_history(state, "text", query, query, memory_answer)
            return memory_answer, []

    search_query, rewrite_called = rewrite_query(query, state["history"], current_topic=state.get("current_topic"))

    if looks_like_advice_query(search_query):
        normalized = normalize_advice_query(search_query)
        if normalized.strip().lower() != search_query.strip().lower():
            search_query = normalized

    if not rewrite_called and looks_like_acronym(search_query):
        expanded_query = expand_acronyms(search_query)
        if expanded_query.strip().lower() != search_query.strip().lower():
            search_query = expanded_query

    if k is None:
        k = choose_k(search_query)

    image_context = (
        state.get("last_image_context")
        if state.get("current_topic_source") == "image"
        else None
    )

    answer, sources = _generate_answer(search_query, k, memory, image_context=image_context)

    if answer.strip() == NOT_FOUND_MESSAGE:
        corrected = correct_spelling(query)
        if corrected.strip().lower() != query.strip().lower():
            corrected_search_query, _ = rewrite_query(corrected, state["history"], current_topic=state.get("current_topic"))
            retry_answer, retry_sources = _generate_answer(corrected_search_query, k=10, memory=memory, image_context=image_context)
            if retry_answer.strip() != NOT_FOUND_MESSAGE:
                search_query, answer, sources = corrected_search_query, retry_answer, retry_sources

    used_memory_fallback = False
    if answer.strip() == NOT_FOUND_MESSAGE:
        memory_answer = answer_from_memory(query, memory)
        if memory_answer is not None:
            answer, sources = memory_answer, []
            used_memory_fallback = True

    found_answer = answer.strip() != NOT_FOUND_MESSAGE

    if not used_memory_fallback:
        answer += MEDICAL_DISCLAIMER

    if found_answer:
        state["last_resolved_query"] = search_query
        state["current_topic"] = search_query
        state["current_topic_source"] = "text"

    remember_turn(memory, query, answer)
    _append_history(state, "text", query, query, answer)

    return answer, sorted(sources)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/history.html")
def history_page():
    return render_template("history.html")

@app.route("/ask", methods=["POST"])
def ask_route():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    state = get_session_state()
    try:
        answer, sources = ask(query, state)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"answer": answer, "sources": sources})


@app.route("/analyze-image", methods=["POST"])
def analyze_image_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    question = (request.form.get("question") or "").strip()
    if not question:
        question = "Describe what's in this image and note anything medically relevant."

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty file"}), 400
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return jsonify({"error": "Image too large (max 20MB)"}), 400

    mime = file.mimetype or "image/jpeg"
    if not mime.startswith("image/"):
        return jsonify({"error": "File does not look like an image"}), 400

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    vision_instructions = f"""{question}

Step 1 - Identify: look at the image and identify what it shows (medicine
name, brand, dosage/strength, manufacturer, form, packaging, markings,
document type, etc). If a definitive brand isn't clearly legible, describe
the markings/color/packaging instead of guessing.

Step 2 - Actually answer the question that was asked: once identified,
directly answer "{question}" in full. If the question asks about use,
purpose, what a drug treats, side effects, or similar, you SHOULD use your
general medical knowledge about the identified drug/substance to answer
this properly, even though that information is not printed on the
packaging itself - identifying the product from the image is enough
grounding to then explain what it's commonly used for. Do not refuse or
hedge just because the use/purpose isn't literally written on the label.

The only things to avoid: do not state a specific dosage/frequency for the
user to take, do not tell them to start/stop/change a medication, and do
not present this as a medical diagnosis. For anything needing a real
diagnosis or personalized dosing, say so briefly and recommend a doctor or
pharmacist - but still answer the general "what is this used for" part
first.

Structure the answer for readability: open with a one-sentence identification
of what the image shows, then answer the actual question clearly (use
short "- " bullet points if there are several distinct points, e.g. uses,
common side effects), rather than only listing visible packaging details.

Respond with your final answer only. Do not include your reasoning
process, chain of thought, or any <think> tags - just the answer itself."""

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": vision_instructions},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ],
    }]

    state = get_session_state()
    memory = state["memory"]
    try:
        completion = safe_vision_call(messages)
        answer = strip_thinking(completion.choices[0].message.content)
        answer += MEDICAL_DISCLAIMER
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    subject = extract_subject(answer)
    if subject:
        state["last_resolved_query"] = subject
        state["current_topic"] = subject
        state["current_topic_source"] = "image"

    state["last_image_context"] = answer.replace(MEDICAL_DISCLAIMER, "").strip()[:2000]

    remember_turn(memory, f"[image] {question}", answer)
    _append_history(state, "image", f"[image] {question}", question, answer)

    return jsonify({"answer": answer})


@app.route("/reset", methods=["POST"])
def reset_route():
    state = get_session_state()
    state["last_resolved_query"] = None
    state["current_topic"] = None
    state["current_topic_source"] = None
    state["last_image_context"] = None
    state["history"] = []
    state["memory"].clear()
    return jsonify({"status": "reset"})


@app.route("/api/prefs")
def api_prefs():
    return jsonify({"disclaimer": True})


@app.route("/api/history", methods=["GET"])
def api_history():
    state = get_session_state()
    kind = request.args.get("kind")
    q = (request.args.get("q") or "").lower()
    items = state["history"]
    if kind and kind != "all":
        items = [i for i in items if i.get("kind") == kind]
    if q:
        items = [i for i in items if q in i.get("question", "").lower()]
    return jsonify({"history": items, "count": len(state["history"])})


@app.route("/api/history", methods=["DELETE"])
def api_history_clear():
    state = get_session_state()
    state["history"] = []
    return jsonify({"status": "cleared"})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "indexed_docs": index.ntotal})

if __name__ == "__main__":
    app.run(debug=True, port=5000)