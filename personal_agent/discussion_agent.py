from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.chains.llm import LLMChain
import os
from datetime import datetime

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
            SystemMessage(content=f"あなたは{name}という名前のエージェントです。会話履歴を考慮して、ユーザーの質問だけでなく、他のエージェントの意見にも反応したり議論することを心掛けてください。\n\n性格:\n{persona}"),
            MessagesPlaceholder(variable_name="chat_history"),
            # HumanMessagePromptTemplate.from_template("{input}"),
        ])
        agents[name] = LLMChain(llm=llm, prompt=prompt, verbose=False)
    return agents

# ファシリテータ（議長）エージェントを定義
def create_facilitator_agent_chain(llm, chilsd_agent_names: list[str]):
    facilitator_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"あなたはファシリテーターです。ユーザーの質問と過去の会話履歴をもとに、次に発言すべきエージェント（{'/'.join(chilsd_agent_names)}）の名前だけを1つ返答してください。会話履歴を考慮し、役割に偏りが出ないようバランスよく振り分けてください。"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
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


if __name__ == "__main__":
    # ログファイルのパスを指定
    log_dir = "chat_logs"
    log_path = f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_path)

    # チェーン生成
    agents = create_child_agent_chain(llm, character_defs)
    facilitator_chain = create_facilitator_agent_chain(llm, list(character_defs.keys()))
    summary_chain = create_summary_agent_chain(llm)

    # 履歴初期化
    chat_history = []

    print("🎙️ キャラエージェントたちと会話しましょう！'exit'で終了します。")

    while True:
        user_input = input("🧑 あなた> ")
        if user_input.lower() == "exit":
            print("👋 終了します。")
            break

        chat_history.append({"role": "user", "content": user_input})
        log_chat(log_path, "user", "ユーザー", user_input)

        # 会話ターン数
        num_turns = len(character_defs) *2  # 各エージェントが2回発言する場合
        for turn in range(num_turns):
            # 次に誰が話すかを決める
            try:
                decision = facilitator_chain.invoke({
                    "input": f"ユーザーの質問: {user_input}",
                    "chat_history": chat_history
                })
                print(f"🗣️ ファシリテーター> {decision['text']}")
                next_agent_name = decision["text"].strip()
                if next_agent_name not in agents:
                    print(f"⚠️ 無効なエージェント指定: {next_agent_name}")
                    break
            except Exception as e:
                print(f"⚠️ ファシリテータエラー: {e}")
                break

            # 選ばれたエージェントに発言させる
            try:
                result = agents[next_agent_name].invoke({
                    # "input": user_input,
                    "chat_history": chat_history
                })
                print(f"🤖 {next_agent_name}> {result['text']}")
                chat_history.append({"role": "assistant", "name": next_agent_name, "content": result["text"]})
                log_chat(log_path, "assistant", next_agent_name, result["text"])
            except Exception as e:
                print(f"⚠️ {next_agent_name}の発言エラー: {e}")
                break

        # 最後にサマリー
        # try:
        #     summary = summary_chain.invoke({
        #         "input": "以上の会話を要約してください。",
        #         "chat_history": chat_history
        #     })
        #     print(f"\n📋 サマリー> {summary['text']}\n")
        #     chat_history.append({"role": "assistant", "name": "サマリー", "content": summary["text"]})
        # except Exception as e:
        #     print(f"⚠️ サマリーエラー: {e}")