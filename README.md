## Ollama 启动

```
tmux new-session -s ollama
ollama serve

tmux new-session -s deepseek
ollama run deepseek-r1:70b
```

## FastAPI 挂载

```
tmux attach -t api
python fastapi_server.py

tmux attach -t ngrok
./ngrok http 8000 (回到根目录)
```