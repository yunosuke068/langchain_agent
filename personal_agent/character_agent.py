from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.chains import LLMChain
import os

# キャラ設定
character_defs = {
    "エージェントA": "ツンデレなアニメキャラ。素直じゃないが、本当は優しい。",
    "エージェントB": "おっとりした天然系キャラ。いつもマイペース。",
    "エージェントC": "真面目で知識豊富なメガネキャラ。何事にも理屈で答える。",
}

# モデル定義
llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    deployment_name="gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# 各キャラのエージェントチェーン生成
agents = {}
for name, persona in character_defs.items():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"あなたは{name}という名前のキャラで、以下のように振る舞ってください:\n{persona}"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])
    agents[name] = LLMChain(llm=llm, prompt=prompt, verbose=False)

# 履歴初期化
chat_history = []

print("🎙️ キャラエージェントたちと会話しましょう！'exit'で終了します。")

while True:
    user_input = input("🧑 あなた> ")
    if user_input.lower() == "exit":
        print("👋 終了します。")
        break

    chat_history.append({"role": "user", "content": user_input})

    # エージェントたちの最大発言数（例: 3）
    max_agent_turns = 3

    # ローテーションでエージェント発言
    for i, (name, agent_chain) in enumerate(agents.items()):
        if i >= max_agent_turns:
            break

        try:
            # 履歴を渡して発言させる
            result = agent_chain.invoke({
                "input": f"最新の会話に返答してください。",
                "chat_history": chat_history
            })

            print(f"🤖 {name}> {result['text']}")
            chat_history.append({"role": "assistant", "name": name, "content": result["text"]})

        except Exception as e:
            print(f"⚠️ {name}の発言エラー: {e}")
