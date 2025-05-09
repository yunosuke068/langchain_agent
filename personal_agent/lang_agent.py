from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
import os

# 1. モデル定義
# llm = ChatOpenAI(model="gpt-4", temperature=0)
llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    deployment_name="gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# 2. Tool定義（デコレータ方式）
@tool
def add_numbers(x: float, y: float) -> float:
    """2つの数値を足し算します。"""
    return x + y

tools = [add_numbers]

# 3. Promptを定義
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="あなたはユーザーの質問に答えるAIエージェントです。必要に応じてツールを使用してください。"),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 4. Agent作成
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# 5. 実行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # 6. 実行
# response = agent_executor.invoke({
#     "input": "4と5を足して",
#     "chat_history": []  # historyがある場合はここに追加
# })
# print(response["output"])

# === チャット履歴を保持 ===
chat_history = []

# === コンソールチャットループ ===
print("💬 エージェントと会話できます。'exit'で終了。")
while True:
    user_input = input("🧑 あなた> ")

    if user_input.lower() in ["exit", "quit"]:
        print("👋 終了します。")
        break

    try:
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        response = result["output"]
        print(f"🤖 エージェント> {response}")

        # 会話履歴に追加
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})

    except Exception as e:
        print(f"⚠️ エラー: {e}")