import os
import traceback
import logging
from datetime import datetime
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains.llm import LLMChain
from dataclasses import dataclass
import json
import streamlit as st

# ========== LLM 初期化 ==========
llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    deployment_name="gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# ========== 共通脳ツール群 ==========
class LLMBrainTool:
    name: str
    description: str
    prompt: str

    def __init__(self, name: str, description: str, prompt: str):
        self.name = name
        self.description = description
        self.prompt = prompt

    def __call__(self, input_text: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"あなたは{self.name}です。{self.prompt}"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        chain = prompt | llm
        content = chain.invoke({"input": input_text}).content
        logging.info(f"Tool {self.name} invoked with input: {input_text}\nOutput: {content}")
        return content

brain_functions = {
    "意図推論ツール": LLMBrainTool(
        name="意図推論ツール",
        description="ユーザーの意図を推論し、目的を明確にします。",
        prompt="あなたの役割はユーザーの発言や会話履歴を読み取り、その人が何を目的としてチャットをしているのかを推論することです。"
    ),
    "課題分解ツール": LLMBrainTool(
        name="課題分解ツール",
        description="問題を要素に分解し、どの順序や観点で考えるべきかを構造化します。",
        prompt="あなたの役割は与えられた問題を複数の要素に分解し、それぞれをどのような順序や視点で考えるべきかを整理することです。"
    ),
    "背景知識整理ツール": LLMBrainTool(
        name="背景知識整理ツール",
        description="自分が持つ知識の中から関係があるものを取り出し、課題に関連づけます。",
        prompt="あなたの役割は自分の知識の中から、与えられた課題に関連する背景知識を抽出し、どのように関係するかを整理することです。"
    ),
    "ステップ計画ツール": LLMBrainTool(
        name="ステップ計画ツール",
        description="解決までの手順や議論の流れを組み立て、計画を立てます。",
        prompt="あなたの役割は課題を解決するために必要なステップを計画し、議論や作業の流れを明確にすることです。"
    ),
    "自己評価・矛盾検出ツール": LLMBrainTool(
        name="自己評価・矛盾検出ツール",
        description="自分の推論・答えに矛盾がないかをチェックし、必要に応じて修正を促します。",
        prompt="あなたの役割は自分の推論や回答内容を再評価し、矛盾や誤りがないかを確認し、問題があれば修正を提案することです。"
    ),
    "最終出力整形ツール": LLMBrainTool(
        name="最終出力整形ツール",
        description="自分の答えを、相手（ユーザーや他のエージェント）にわかりやすい形にまとめ、整形します。",
        prompt="あなたの役割は自分が出した結論や提案を、相手が理解しやすい形にまとめ、明確に整形して伝えることです。出力は会話の一部として自然な形で行ってください。例えば、JSON形式ではなく、自然な言葉で発言してください。"
    )
}
brain_tools = []
for name, tool in brain_functions.items():
    brain_tools.append(
        Tool(
            name=name,
            func=lambda x: tool(x),
            description=tool.description
        )
    )

# ========== 各専門固有ツール ==========
def create_legal_tools(llm):
    # 例：法律データベース検索
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="あなたは法律専門家です。質問に関連する条文や判例を参照します。"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    chain = prompt | llm
    return [
        Tool(
            name="法律DB検索", 
            func=lambda x: chain.invoke({"input": x}).content, 
            description="法律の条文や判例を調べる"
        )
    ]

def create_engineer_tools(llm):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="あなたはエンジニアです。技術仕様を評価し、実装案を考えます。"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    chain = prompt | llm
    return [
        Tool(
            name="技術仕様評価", 
            func=lambda x: chain.invoke({"input": x}).content, 
            description="技術的実現性や工数を見積もる"
        )
    ]

def create_common_sense_tools(llm):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="あなたは一般常識に詳しい専門家です。世間的な感覚や常識を反映します。"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    chain = prompt | llm
    return [
        Tool(
            name="世間感覚分析", 
            func=lambda x: chain.invoke({"input": x}).content, 
            description="一般的な人々の受け止め方を予測する"
        )
    ]

# ========== 専門エージェント ==========
def create_specialist_agent(name: str, system_msg:str, specific_tools:list):
    all_tools = brain_tools + specific_tools
    return initialize_agent(
        tools=all_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        system_message=f"あなたの名前は{name}です。\n{system_msg}。ユーザーとの会話を通じて、あなたの専門知識を活かして答えてください。また自然な会話の流れを意識し、他のエージェントとの議論も行ってください。",
    )

# エージェント定義
agent_defs = {
    "法律エージェント": {
        "name": "法律エージェント",
        "avatar": "https://icooon-mono.com/i/icon_14451/icon_144511_64.png",
        "description": "法律の専門家として、法律に関する質問に答えます。",
        "tool": create_specialist_agent(
            name="法律エージェント", 
            system_msg="あなたは法律の専門家です。", 
            specific_tools=create_legal_tools(llm)
        )
    },
    "エンジニアエージェント": {
        "name": "エンジニアエージェント",
        "avatar": "https://icooon-mono.com/i/icon_10193/icon_101931_64.png",
        "description": "技術の専門家として、技術的な質問に答えます。",
        "tool": create_specialist_agent(
            name="エンジニアエージェント", 
            system_msg="あなたは技術の専門家です。", 
            specific_tools=create_engineer_tools(llm)
        )
    },
    "一般常識エージェント": {
        "name": "一般常識エージェント",
        "avatar": "https://icooon-mono.com/i/icon_11127/icon_111271_64.png",
        "description": "一般常識の専門家として、世間的な感覚や常識を反映します。",
        "tool": create_specialist_agent(
            name="一般常識エージェント", 
            system_msg="あなたは一般常識の専門家です。", 
            specific_tools=create_common_sense_tools(llm)
        )
    }
}


# ========== 専門エージェントの動作テスト ==========
# user_question = "新しい社員研修制度を設計する際の重要な注意点は何ですか？"
# datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
# log_dir = "chat_logs"

# os.makedirs(log_dir, exist_ok=True)

# legal_result = legal_agent.invoke(input=user_question)
# engineer_result = engineer_agent.invoke(input=user_question)
# common_result = common_sense_agent.invoke(input=user_question,config={"callbacks": [LogCallbackHandler(f"{log_dir}/{datetime_str}_common_agent.txt")]})

# print("\n=== 各専門家の回答 ===\n")
# print("法律専門家の回答:", legal_result)
# print("エンジニア専門家の回答:", engineer_result.content)
# print("一般常識専門家の回答:", common_result)


# ========== ファシリテータ（議長）エージェントを定義　==========
def create_facilitator_prompt(chilsd_agent_names: list[str]):
    facilitator_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"あなたはファシリテーターです。ユーザーの質問と過去の会話履歴をもとに、次に発言すべきエージェント（{'/'.join(chilsd_agent_names)}）の名前だけを1つ返答してください。会話履歴を考慮し、役割に偏りが出ないようバランスよく振り分けてください。エージェント名以外の情報は返さないでください。"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    return facilitator_prompt


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
    st.session_state.agents = agent_defs
if "facilitator" not in st.session_state:
    st.session_state.facilitator_prompt = create_facilitator_prompt(list(agent_defs.keys()))

# 入力欄
user_input = st.chat_input("あなたの質問を入力...")

# 過去の会話を表示
for msg in st.session_state.display_chat_history:
    with st.chat_message(name=msg.get("name", msg["role"]), avatar=msg.get("avatar", None)):
        st.markdown(msg.get("name", msg["role"]) + ":")
        st.markdown(msg["content"])

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.display_chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.agent_history = []  # エージェントの発言履歴を初期化

    # エージェントの発言ターン数
    for _ in range(len(agent_defs)):
        try:
            prompt_value = st.session_state.facilitator_prompt.invoke({
                    "input": f"{user_input}",
                    "chat_history": st.session_state.chat_history
                })

            decision = llm.invoke(prompt_value)
            print(f"🗣️ ファシリテーター> {decision}")
            next_agent_name = decision.content.strip()
            if next_agent_name not in st.session_state.agents:
                st.warning(f"⚠️ 無効なエージェント指定: {next_agent_name}")
                break

            st.session_state.agent_history.append(next_agent_name)
            
            result = agent_defs[next_agent_name]["tool"].invoke(
                f"ユーザー発言: {user_input}\n過去の会話履歴: {st.session_state.chat_history}",
            )
            print(f"🤖 {next_agent_name}> {result}")

            content = result["output"]

            if content != "無効なレスポンス":
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "name": next_agent_name,
                    "content": f"{{'name': '{next_agent_name}', 'content': '{content}'}}"
                })
            st.session_state.display_chat_history.append({
                "role": "assistant",
                "avatar": agent_defs[next_agent_name]["avatar"],
                "name": next_agent_name,
                "content": content
            })
            with st.chat_message(name=next_agent_name, avatar=agent_defs[next_agent_name]["avatar"]):
                st.markdown(next_agent_name + ":")
                st.markdown(content)

        except Exception as e:
            # st.error(f"エラー: {e}")
            # break
            error_message = traceback.format_exc()

            st.warning(f"⚠️ {next_agent_name}の発言エラー: {error_message}")


# streamlit run .\personal_agent\multi_zero_shot_agent_streamlit.py