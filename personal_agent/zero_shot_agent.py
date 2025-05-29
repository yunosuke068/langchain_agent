# 必要なライブラリ
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.tools import Tool
import os, logging

logging.basicConfig(level=logging.INFO)

# LLMの初期化
llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    deployment_name="gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# エージェント設定
agent_definitions = {
    "分解係": {
        "description": "タスクを分解して明確な問いに整理します。",
        "system_prompt": "あなたは与えられたタスクを複数の問いに分解し、論点を明確にする専門家です。"
    },
    "検索係": {
        "description": "必要な外部情報（契約前に作業を開始したために発生したトラブル事例を通じて、正式な契約書の重要性とモデル契約書の活用方法が含まれるDB）を収集して提供します。",
        "system_prompt": "あなたは与えられた問いに対し、外部情報（仮にWeb検索結果があると想定）をもとに回答する情報収集の専門家です。"
    },
    "検証係": {
        "description": "他のエージェントの意見に対して矛盾や誤りを検証します。",
        "system_prompt": "あなたは他のエージェントの意見を精査し、論理的な矛盾や誤解がないかをチェックする検証係です。批判的思考を重視してください。"
    },
    "アイデア係": {
        "description": "創造的な発想や代替案を提案します。",
        "system_prompt": "あなたは創造的なアイデアを出す役割です。常識にとらわれず、新しい視点やユニークな発想を出してください。"
    },
    "まとめ係": {
        "description": "他の意見を統合し、最終的な結論を出します。",
        "system_prompt": "あなたは他のエージェントの意見を要約し、バランスの取れた結論を導き出す役割です。"
    }
}


import requests

search_url = "http://localhost:8000/faiss/deep_test/search"  # 仮の外部APIエンドポイント

def external_search_api(
    query: str,
    logger: logging.Logger = logging.getLogger(__name__),
) -> str:
    # 仮：外部API呼び出し
    response = requests.post(search_url, json={"query": query, "top_k": 5})
    if response.status_code == 200:
        data = response.json()
        summaries = f"{data}"

        logger.info(f"検索結果: {summaries}")

        # ここでLLM要約を入れる
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="以下の検索結果を、ユーザーの質問に関連するポイントを絞って要約してください。"),
            HumanMessagePromptTemplate.from_template("ユーザー質問: {query}\n検索結果:\n{summaries}")
        ])
        summary_chain = prompt | llm
        summary = summary_chain.invoke({"query": query, "summaries": summaries}).content

        return summary

    else:
        return "APIリクエストが失敗しました。"


# print("検索テスト:", external_search_api("トラブル事例"))


# 各エージェントをToolに変換
agent_tools = []
for name, cfg in agent_definitions.items():
    if name == "検索係":
        # 検索係は外部APIを直接呼ぶ
        tool = Tool(
            name=name,
            func=external_search_api,
            description=cfg["description"]
        )
    else:
        # 他の係はLLMChainを使う
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=cfg["system_prompt"]),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        chain = prompt | llm

        def make_tool(chain):
            return Tool(
                name=name,
                func=lambda x: chain.invoke({"input": x}).content,
                description=cfg["description"]
            )

        tool = make_tool(chain)

    agent_tools.append(tool)


from langchain.agents import initialize_agent, AgentType

agent_executor = initialize_agent(
    tools=agent_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 実行
result = agent_executor.invoke(
    # input = "新しい社員研修制度を考えたい。どう整理し、考え始めるべき？"
    input="トラブル事例を調べてまとめて"
)
print(result["output"])