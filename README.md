### ビルド
```bash
docker compose -f .\docker-compose.personal_agent.yml up -d --build
```

### シェルに入る
```bash
docker compose exec personal-agent bash
```

### ディスカッションの実行
```bash
python discussion_agent.py
```