import asyncio
import json
import uuid

from websockets.asyncio.client import connect as ws_connect

async def iter_word():
    for w in ["你好","我是捞捞","请问有什么可以帮您"]:
        yield w

async def test():
    async with ws_connect(
            "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
            additional_headers={
                "Authorization": "bearer sk-94072328cdfd4e2bb063f450c7564387",
            },
    ) as ws:
        task_id = uuid.uuid4().hex
        start_param = {
            "header": {
                "action": "run-task",
                "task_id": task_id,
                "streaming": "duplex",
            },
            "payload": {
                "task_group": "audio",
                "task": "tts",
                "function": "SpeechSynthesizer",
                "model": "cosyvoice-v3-flash",
                "parameters": {
                    "text_type": "PlainText",
                    "voice": "longanyang",
                    "format": "pcm",
                    "sample_rate":16000,
                    "volume": 50,
                    "rate": 1,
                    "pitch": 1,
                },
                "input": {},
            },
        }
        await ws.send(json.dumps(start_param))

        async def send_task(wsclient) -> None:
            """等待 task-started，然后把 LLM 文本逐块通过 continue-task 发送，
            遇到 FlushSentinel（大模型回复结束）或 _input_ch 关闭后发 finish-task。"""
            async for data in iter_word():
                if isinstance(data, str):
                    await wsclient.send(json.dumps({
                        "header": {
                            "action": "continue-task",
                            "task_id": task_id,
                            "streaming": "duplex",
                        },
                        "payload": {"input": {"text": data}},
                    }))
                    print("[AliTTS] 发送文本片段: %s", data)
                    await asyncio.sleep(1)

            await wsclient.send(json.dumps({
                "header": {
                    "action": "finish-task",
                    "task_id": task_id,
                    "streaming": "duplex",
                },
                "payload": {"input": {}},
            }))
            print("[AliTTS] finish-task 已发送")

        async def recv_task() -> None:
            """接收服务端消息：二进制数据推入 emitter，task-finished 时结束。"""
            while True:
                response = await ws.recv()

                if isinstance(response, bytes):
                    print("收到生成的音频字节")
                    continue

                msg = json.loads(response)
                event = msg.get("header", {}).get("event", "")

                if event == "task-started":
                    print("[AliTTS] task-started")
                    asyncio.get_event_loop().create_task(send_task(ws))
                elif event == "task-finished":
                    print("[AliTTS] task-finished，连接关闭")
                    break
                elif event == "task-failed":
                    print("[AliTTS] task-failed: %s", json.dumps(msg))
                    break
        tasks = [
            asyncio.create_task(recv_task()),
        ]
        await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(test())

