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
        prompt="あなたの役割は自分が出した結論や提案を、相手が理解しやすい形にまとめ、明確に整形して伝えることです。"
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
def create_specialist_agent(name, system_msg, specific_tools):
    all_tools = brain_tools + specific_tools
    return initialize_agent(
        tools=all_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        system_message=system_msg
    )

legal_agent = create_specialist_agent("法律エージェント", "あなたは法律の専門家です。", create_legal_tools(llm))
engineer_agent = create_specialist_agent("エンジニアエージェント", "あなたは技術の専門家です。", create_engineer_tools(llm))
common_sense_agent = create_specialist_agent("一般常識エージェント", "あなたは一般常識の専門家です。", create_common_sense_tools(llm))

# ========== ログ設定 ==========
from langchain.callbacks.base import BaseCallbackHandler

class LogCallbackHandler(BaseCallbackHandler):
    def __init__(self, log_path):
        self.log_path = log_path

    def on_agent_action(self, action, **kwargs):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"<on_agent_action start> " + "=" * 20 + "\n")
            f.write(f"[Thought] {action.log}\n")
            f.write(f"[Action] {action.tool}\n")
            f.write(f"[Action Input] {action.tool_input}\n")
            f.write("-" * 40 + f" <on_agent_action end>" + "\n")

    def on_tool_end(self, output, **kwargs):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[Observation] {output}\n")
            f.write("-" * 40 + "\n")

    def on_chain_end(self, outputs, **kwargs):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[Final Output] {outputs}\n")
            f.write("=" * 60 + "\n")


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
        SystemMessage(content=f"あなたはファシリテーターです。ユーザーの質問と過去の会話履歴をもとに、次に発言すべきエージェント（{'/'.join(chilsd_agent_names)}）の名前だけを1つ返答してください。会話履歴を考慮し、役割に偏りが出ないようバランスよく振り分けてください。"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    return facilitator_prompt


# ========== シェルベースチャットボット ==========
if __name__ == "__main__":
    # ログファイルのパスを指定
    log_dir = "chat_logs"
    log_path = f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_path)

    # チェーン生成
    agent_defs = {
        "法律エージェント": {
            "name": "法律エージェント",
            "description": "法律の専門家として、法律に関する質問に答えます。",
            "tool": legal_agent
        },
        "エンジニアエージェント": {
            "name": "エンジニアエージェント",
            "description": "技術の専門家として、技術的な質問に答えます。",
            "tool": engineer_agent
        },
        "一般常識エージェント": {
            "name": "一般常識エージェント",
            "description": "一般常識の専門家として、世間的な感覚や常識を反映します。",
            "tool": common_sense_agent
        }
    }
    # facilitator_chain = create_facilitator_agent_chain(llm, list(agent_defs.values()))
    facilitator_prompt = create_facilitator_prompt(list(agent_defs.keys()))

    # 履歴初期化
    chat_history = []

    print("🎙️ キャラエージェントたちと会話しましょう！'exit'で終了します。")

    while True:
        user_input = input("🧑 あなた> ")
        if user_input.lower() == "exit":
            print("👋 終了します。")
            break

        chat_history.append({"role": "user", "content": user_input})
        # log_chat(log_path, "user", "ユーザー", user_input)

        # 会話ターン数
        num_turns = len(agent_defs) *1  # 各エージェントが2回発言する場合
        for turn in range(num_turns):
            # 次に誰が話すかを決める
            try:
                
                prompt_value = facilitator_prompt.invoke({
                    "input": "ユーザーの質問",
                    "chat_history": chat_history
                })

                decision = llm.invoke(prompt_value)
                print(f"🗣️ ファシリテーター> {decision.content}")
                next_agent_name = decision.content.strip()
                if next_agent_name not in agent_defs:
                    print(f"⚠️ 無効なエージェント指定: {next_agent_name}")
                    break
            except Exception as e:
                print(f"⚠️ ファシリテータエラー: {e}\n\n{traceback.format_exc()}")
                break

            # 選ばれたエージェントに発言させる
            try:
                result = agent_defs[next_agent_name]["tool"].invoke(
                    f"ユーザー発言: {user_input}\n過去の会話履歴: {chat_history}",
                )
                print(f"🤖 {next_agent_name}> {result['output']}")
                chat_history.append({"role": "assistant", "name": next_agent_name, "content": next_agent_name+": "+result["output"]})
                # log_chat(log_path, "assistant", next_agent_name, result["text"])
            except Exception as e:
                print(f"⚠️ {next_agent_name}の発言エラー: {e}")
                break