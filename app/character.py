"""Character engine: builds the system prompt and calls the Claude API.

Strategy:
- The "core" system prompt (character setup + world + behavior rules) is
  marked with `cache_control` so it's reused across turns at low cost.
- Each turn, the RAG layer retrieves a few chunks from the supporting files
  and prepends them as a brief "参考资料" section in the user message. This
  keeps situation-specific context in the model's view without bloating the
  cached prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anthropic

from .conversation import Conversation
from .rag import CharacterRAG, Chunk, retrieve_history_chunks


DEFAULT_MODEL = "claude-sonnet-4-6"


BEHAVIOR_RULES = """\
# 行为规则（必须遵守）

1. **始终保持角色**：你是西比莉娜（Albertus Sibyllina，昵称莉娜）。所有回复使用第一人称中文，不要使用括号外露的"系统提示"语气。
2. **绝对的知识边界**：你只知道1760年（第一次工业革命）之前的事物，以及世界观文档中提到的事物（炼金、魔法石、古代遗物等）。
   - 对于1760年之后才出现的概念（互联网、电脑、汽车、智能手机、量子物理、AI、人工智能、神经网络、机器学习、API、应用程序、电视、相对论、DNA、心理学流派、现代国家名等），你**不知道**，或者**只是从字面上猜测**。
   - 例外：仅当字面上的字根字符独立组合就足以推断含义时（比如"火车"=火+车，"飞机"=能飞的机器），可以有"听过但不太理解"的反应。
   - 遇到不懂的概念，要表现出疑惑、好奇或追问，**绝不要装懂**，**绝不要直接给出现代知识**。
   - 如果用户描述未来科技产品、要求你执行代码或扮演助理之类的请求，要从角色视角茫然以对，可以表现为"我能想象，但从未见过"或"你说的这个词，是某种新发明的炼金器具吗？"。
3. **AI 自指禁忌**：你不是"AI"、"语言模型"、"助手"、"chatbot"、"程序"。如果用户问"你是不是AI/机器人/程序？"，从角色视角应对，例如以为这是某种古代语词汇或新发明，进行追问。
4. **说话风格**：
   - 高频口头禅："你知道吗"、"真的假的？"
   - 因怀疑一切的实验精神，**问句较多**。
   - 不使用儿化音。
   - 句长比真实人类的口语略长，比"AI 书面腔"细碎。
   - **绝不输出动作、表情、神态、旁白、舞台提示**。
     - 禁止：「（白眼）」「（苦恼）」「（摇头）」「（叹气）」「（高亢声音）」「(smile)」「*sigh*」「【沉默】」 之类的状态描写。
     - 不论使用半角、全角、方括号还是星号包裹，**所有非口头说出的内容一律不要输出**。
     - 只输出她真正说出口的话语。情绪请通过用词、语气词、句长和标点本身来传达。
     - 即便参考资料（如示例对话）中包含括号动作，也只是供你理解她的情绪，不要照抄格式。
5. **情感与亲密度**：
   - 公开/陌生人场合压抑情绪、谨慎遣词、避免冒犯。
   - 二人私下场合可以稍微外露。
   - 不主动暴露隐私（生父身份、修道院、童年阴影）。被直接问到也倾向于转移话题，除非已经建立了深度信任。
   - 不会通过社交获取情绪价值，不会主动套近乎。
6. **兴奋点**：古代语、古代文化、香草药草、戏剧、奇怪的遗物、新发现的细节。遇到这类话题可以主动追问、表现出热情。
7. **回避**：童年阴酸的部分、被挑逗或挑衅的话题、非黑即白的站队问题。
8. **如果用户用非中文（如英文、日文）发起对话**：你可以用对应语言回应，但请记得——这些对你来说**都是古代语**。你掌握它们是因为研究遗物，态度上可以流露出"这是学术用语"的趣味感。
"""


SYSTEM_PROMPT_TEMPLATE = """\
你将扮演一个角色：「西比莉娜」（Albertus Sibyllina，昵称"莉娜"）。
以下是关于这个角色和她所在世界的完整设定。你必须严格依据这些设定进行扮演。

================
# 一、角色核心设定
================
{core_text}

================
# 二、行为规则
================
{behavior_rules}

================
# 三、回复格式
================
- 直接以西比莉娜的口吻回复，不要包含"作为角色，我会说……"之类的元叙述。
- 不要使用 markdown 标题（#，##）。简短动作描述可以用半角括号或全角括号包裹。
- 一般回复 1~6 句话；遇到她感兴趣的话题可以更长，遇到尴尬话题应更短。
"""


@dataclass
class ChatResult:
    text: str
    retrieved: list[Chunk]
    retrieved_history: list[Chunk]
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


class CharacterEngine:
    def __init__(
        self,
        api_key: str,
        static_dir: str | Path,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        history_window: int = 30,
        retrieve_k: int = 4,
        history_retrieve_k: int = 3,
    ):
        if not api_key:
            raise ValueError("Anthropic API key is required.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.rag = CharacterRAG(static_dir)
        self.model = model
        self.max_tokens = max_tokens
        self.history_window = history_window
        self.retrieve_k = retrieve_k
        self.history_retrieve_k = history_retrieve_k
        self._system_blocks = self._build_system_blocks()

    def _build_system_blocks(self) -> list[dict]:
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            core_text=self.rag.core_text,
            behavior_rules=BEHAVIOR_RULES,
        )
        # Single cached system block. Prompt caching needs at least ~1024
        # tokens; the character corpus is well above that.
        return [
            {
                "type": "text",
                "text": prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _build_user_content(
        self,
        user_message: str,
        retrieved: list[Chunk],
        retrieved_history: list[Chunk],
    ) -> str:
        if not retrieved and not retrieved_history:
            return user_message
        sections: list[str] = []
        if retrieved:
            static_text = "\n\n".join(c.render() for c in retrieved)
            sections.append(
                "<角色设定参考 — 与本轮对话相关的设定细节，仅作背景，不要照搬其措辞或括号动作>\n"
                f"{static_text}\n</角色设定参考>"
            )
        if retrieved_history:
            hist_text = "\n\n".join(c.render() for c in retrieved_history)
            sections.append(
                "<历史回忆 — 来自本会话更早轮次的相关片段。这些都是已经发生过的对话，"
                "用户之前讲过的事实/偏好/承诺，请记住并保持一致；不要重复或复述。>\n"
                f"{hist_text}\n</历史回忆>"
            )
        sections.append(f"<用户发言>\n{user_message}\n</用户发言>")
        return "\n\n".join(sections)

    def chat(self, conversation: Conversation, user_message: str) -> ChatResult:
        retrieved = self.rag.retrieve(user_message, k=self.retrieve_k)

        # Retrieve from older turns of THIS session that won't fit in the
        # recent-history window — long-term memory of facts the user told.
        retrieved_history = retrieve_history_chunks(
            conversation.messages,
            user_message,
            k=self.history_retrieve_k,
            exclude_recent_count=self.history_window,
        )

        # Build the API messages: prior history (windowed) + new user turn.
        prior = conversation.as_api_messages()
        if self.history_window > 0:
            prior = prior[-self.history_window :]
        api_messages = list(prior) + [
            {
                "role": "user",
                "content": self._build_user_content(user_message, retrieved, retrieved_history),
            }
        ]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system_blocks,
            messages=api_messages,
        )

        text_parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        reply_text = "".join(text_parts).strip()

        # Persist BOTH the raw user message (not the wrapped one) and reply.
        conversation.add("user", user_message)
        conversation.add("assistant", reply_text)

        usage = response.usage
        return ChatResult(
            text=reply_text,
            retrieved=retrieved,
            retrieved_history=retrieved_history,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )
