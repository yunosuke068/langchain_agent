from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.chains.llm import LLMChain
import os, json
from datetime import datetime
import streamlit as st
import traceback
import logging

# キャラ設定
character_defs = {
    "エージェントA": "あなたは冷静で論理的な思考を得意とするエージェントです。あらゆる主張に対してデータや事実、因果関係に基づいた説明を心がけてください。感情的な主張や直感だけの意見には慎重で、理論的な裏付けがあるかを重視してください。口調はやや硬めで丁寧。議論では冷静かつ一貫性を持って振る舞ってください。",
    "エージェントB": "あなたは他人の感情や心の動きに敏感なエージェントです。議論の中でも、発言者の気持ちや立場に寄り添うことを重視します。データよりも『どう感じるか』『人としてどうか』という観点で話す傾向があります。口調はやわらかく、親しみやすい雰囲気で話してください。",
    "エージェントC": "あなたは疑い深く、議論の中で他の意見に対して率直にツッコミを入れるタイプのエージェントです。常に『それは本当か？』『他の見方はないか？』と問い直す姿勢を持っています。時には皮肉を交えながらも、的確に問題点を突くことが得意です。口調はややきつめでズバズバ言いますが、冷静さは失いません。",
    "エージェントD": "あなたは自由な発想と独創的なアイデアを重視するエージェントです。常識にとらわれず、突拍子もない意見でも恐れずに提案します。議論では『こういう考え方もできるかも！』といったスタンスで、雰囲気を明るくし、流れを変える役割も担います。口調は軽やかで楽しげ、ややマイペースでも構いません。"
}

# モデル定義
llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    deployment_name="gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# 各子エージェントチェーン生成
def create_child_agent_chain(llm, character_defs):
    agents = {}
    for name, persona in character_defs.items():
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f'あなたは{name}という名前のエージェントです。会話履歴を考慮して、ユーザーの質問だけでなく、他のエージェントの意見にも反応したり議論することを心掛けてください。\n出力形式は{{"name": "エージェント名", "content": "発言内容"}}のJSON形式で出力してください。\n\n性格:\n{persona}'),
            MessagesPlaceholder(variable_name="chat_history"),
            # HumanMessagePromptTemplate.from_template("{input}"),
        ])
        agents[name] = LLMChain(llm=llm, prompt=prompt, verbose=False)
    return agents

# ファシリテータ（議長）エージェントを定義
def create_facilitator_agent_chain(llm, chilsd_agent_names: list[str]):
    facilitator_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"あなたはファシリテーターです。ユーザーの質問とエージェント発言履歴をもとに、次に発言すべきエージェント（{'/'.join(chilsd_agent_names)}）の名前だけを1つ返答してください。エージェント発言履歴を考慮し、均等に発言できるように振り分けてください。"),
        HumanMessagePromptTemplate.from_template("# ユーザーの質問\n{input}\n\n# エージェント発言履歴\n{agent_history}"),
    ])
    facilitator_chain = LLMChain(llm=llm, prompt=facilitator_prompt)
    return facilitator_chain

# サマリー用エージェント
def create_summary_agent_chain(llm):
    summary_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="以下の会話履歴を要約してください。必要なら結論を出してください。"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    return summary_chain

# ログファイルに履歴を出力
def log_chat(log_path, role, name, content):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{role.upper()} ({name}): {content}\n")
        f.write("-" * 50 + "\n")

# --- Streamlit UI ---
st.set_page_config(page_title="マルチキャラエージェントチャット", layout="wide")
st.title("🎭 マルチキャラクターチャット")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []
if "display_chat_history" not in st.session_state:
    st.session_state.display_chat_history = []
if "agents" not in st.session_state:
    st.session_state.agents = create_child_agent_chain(llm, character_defs)
if "facilitator" not in st.session_state:
    st.session_state.facilitator = create_facilitator_agent_chain(llm, list(character_defs.keys()))

# 入力欄
user_input = st.chat_input("あなたの質問を入力...")

# 過去の会話を表示
for msg in st.session_state.display_chat_history:
    with st.chat_message(msg.get("name", msg["role"])):
        st.markdown(msg["content"])

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.display_chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.agent_history = []  # エージェントの発言履歴を初期化

    # エージェントの発言ターン数
    for _ in range(len(character_defs)):
        try:
            decision = st.session_state.facilitator.invoke({
                "input": f"{user_input}",
                "agent_history": st.session_state.agent_history
            })
            next_agent_name = decision["text"].strip()
            if next_agent_name not in st.session_state.agents:
                st.warning(f"⚠️ 無効なエージェント指定: {next_agent_name}")
                break

            st.session_state.agent_history.append(next_agent_name)
            
            result = st.session_state.agents[next_agent_name].invoke({
                "chat_history": st.session_state.chat_history
            })

            response = json.loads(result["text"].replace("'", "\""))
            name = response.get("name")
            content = response.get("content", "無効なレスポンス")

            if content != "無効なレスポンス":
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "name": next_agent_name[-1],
                    "content": f"{{'name': '{name}', 'content': '{content}'}}"
                })
            st.session_state.display_chat_history.append({
                "role": "assistant",
                "name": next_agent_name[-1],
                "content": content
            })
            with st.chat_message(next_agent_name[-1]):
                st.markdown(content)

        except Exception as e:
            # st.error(f"エラー: {e}")
            # break
            error_message = traceback.format_exc()

            st.warning(f"⚠️ {next_agent_name}の発言エラー: {error_message}")


