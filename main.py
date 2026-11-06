import os
import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("be-na")

app = FastAPI(title="Be Na AI Server")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")

# ESP32 sends signed 16-bit little-endian mono PCM at 16 kHz.
INPUT_SAMPLE_RATE = 16000
# Gemini TTS PCM output is normally 24 kHz. Keep firmware and server in sync.
OUTPUT_SAMPLE_RATE = 24000
MAX_AUDIO_BYTES = 16000 * 2 * 12  # 12 seconds, mono, 16-bit

SYSTEM_PROMPT = """
Bạn là Bé Na, một trợ lý robot thân thiện.
Luôn trả lời bằng tiếng Việt tự nhiên, rõ ràng, ấm áp và ngắn gọn.
Nếu không nghe rõ, hãy nói lịch sự rằng bạn chưa nghe rõ và đề nghị người dùng nói lại.
Không tự nhận là con người. Không bịa thông tin.
"""

ai: Optional[genai.Client] = (
    genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
)

HTML = r"""
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bé Na AI</title>
<style>
body{margin:0;background:#101827;color:#eef2ff;font-family:Arial,sans-serif;padding:20px}
main{max-width:720px;margin:auto}.box{background:#1e293b;border-radius:14px;padding:16px;margin:12px 0}
h1{color:#7dd3fc;text-align:center}button{padding:12px 18px;border:0;border-radius:9px;background:#38bdf8;font-weight:bold}
input{box-sizing:border-box;padding:12px;width:75%;border-radius:8px;border:0}
#answer{color:#86efac;white-space:pre-wrap}#status{color:#fcd34d}
</style>
</head>
<body><main>
<h1>🤖 BÉ NA AI</h1>
<div class="box">Trạng thái: <span id="status">Đang kết nối...</span></div>
<div class="box"><b>Bạn nói:</b><p id="question">...</p></div>
<div class="box"><b>Bé Na trả lời:</b><p id="answer">Xin chào! Mình là Bé Na.</p></div>
<div class="box">
<input id="msg" placeholder="Nhập tin nhắn..." autocomplete="off">
<button onclick="sendText()">Gửi</button>
</div>
</main>
<script>
let ws;
const statusEl=document.getElementById("status");
function connect(){
  const scheme=location.protocol==="https:"?"wss://":"ws://";
  ws=new WebSocket(scheme+location.host+"/ws");
  ws.onopen=()=>statusEl.textContent="Đã kết nối";
  ws.onclose=()=>{statusEl.textContent="Mất kết nối - đang thử lại";setTimeout(connect,3000)};
  ws.onerror=()=>{statusEl.textContent="Lỗi kết nối"};
  ws.onmessage=async(event)=>{
    if(typeof event.data==="string"){
      let d; try{d=JSON.parse(event.data)}catch(e){return}
      if(d.type==="status")statusEl.textContent=d.text||"";
      if(d.type==="question")document.getElementById("question").textContent=d.text||"";
      if(d.type==="answer")document.getElementById("answer").textContent=d.text||"";
    } else {
      // Browser demo receives PCM audio; convert to WAV for playback.
      const buf=await event.data.arrayBuffer();
      const wav=pcmToWav(buf,24000);
      const url=URL.createObjectURL(new Blob([wav],{type:"audio/wav"}));
      const audio=new Audio(url); audio.onended=()=>URL.revokeObjectURL(url); audio.play().catch(()=>{});
    }
  };
}
function sendText(){
  const input=document.getElementById("msg");
  const value=input.value.trim();
  if(value&&ws&&ws.readyState===WebSocket.OPEN){
    ws.send(JSON.stringify({type:"chat",text:value}));
    input.value="";
  }
}
document.getElementById("msg").addEventListener("keydown",e=>{if(e.key==="Enter")sendText()});
function pcmToWav(pcm,sampleRate){
  const header=new ArrayBuffer(44),v=new DataView(header),len=pcm.byteLength;
  const write=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i))};
  write(0,"RIFF");v.setUint32(4,36+len,true);write(8,"WAVE");write(12,"fmt ");
  v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
  v.setUint32(24,sampleRate,true);v.setUint32(28,sampleRate*2,true);
  v.setUint16(32,2,true);v.setUint16(34,16,true);write(36,"data");v.setUint32(40,len,true);
  return new Blob([header,pcm]);
}
connect();
</script></body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML

@app.get("/health")
async def health():
    return {
        "status": "online",
        "gemini_configured": bool(ai),
        "chat_model": CHAT_MODEL,
        "tts_model": TTS_MODEL,
    }

async def generate_text(question: str) -> str:
    if ai is None:
        return "Bé Na chưa được cấu hình khóa GEMINI_API_KEY trên Render."
    response = await asyncio.to_thread(
        ai.models.generate_content,
        model=CHAT_MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=350,
        ),
    )
    return (response.text or "").strip() or "Bé Na chưa tạo được câu trả lời."

async def transcribe_pcm(audio: bytes) -> str:
    if ai is None:
        raise RuntimeError("Chưa cấu hình GEMINI_API_KEY.")
    part = types.Part.from_bytes(
        data=audio,
        mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
    )
    response = await asyncio.to_thread(
        ai.models.generate_content,
        model=CHAT_MODEL,
        contents=[
            "Hãy nghe âm thanh và chép lại chính xác lời người nói bằng tiếng Việt. "
            "Chỉ trả về phần lời nói, không giải thích.",
            part,
        ],
    )
    return (response.text or "").strip()

async def text_to_speech(text: str) -> bytes:
    if ai is None:
        return b""
    response = await asyncio.to_thread(
        ai.models.generate_content,
        model=TTS_MODEL,
        contents=f"Đọc tự nhiên bằng tiếng Việt, giọng nữ thân thiện: {text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=TTS_VOICE
                    )
                )
            ),
        ),
    )
    for candidate in (response.candidates or []):
        content = candidate.content
        if not content or not content.parts:
            continue
        for item in content.parts:
            inline = getattr(item, "inline_data", None)
            if inline and inline.data:
                return inline.data
    logger.warning("Gemini TTS response did not contain inline audio.")
    return b""

async def handle_question(ws: WebSocket, question: str):
    question = question.strip()
    if not question:
        await ws.send_json({"type": "status", "text": "Bé Na chưa nghe thấy nội dung."})
        return
    await ws.send_json({"type": "question", "text": question})
    await ws.send_json({"type": "status", "text": "Bé Na đang suy nghĩ..."})
    answer = await generate_text(question)
    await ws.send_json({"type": "answer", "text": answer})
    await ws.send_json({
        "type": "status",
        "text": "Bé Na đang chuẩn bị giọng nói...",
    })
    try:
        audio = await text_to_speech(answer)
        if audio:
            await ws.send_json({
                "type": "audio",
                "sample_rate": OUTPUT_SAMPLE_RATE,
                "format": "pcm_s16le_mono",
            })
            await ws.send_bytes(audio)
        else:
            await ws.send_json({
                "type": "status",
                "text": "Đã có câu trả lời dạng chữ; chưa nhận được âm thanh TTS.",
            })
    except Exception:
        logger.exception("TTS failed")
        await ws.send_json({
            "type": "status",
            "text": "Có câu trả lời nhưng phần tạo giọng nói bị lỗi.",
        })
    await ws.send_json({"type": "status", "text": "Bé Na đã trả lời. Có thể hỏi tiếp."})

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "status", "text": "Đã kết nối với Bé Na."})
    try:
        while True:
            message = await ws.receive()
            if message.get("text") is not None:
                import json
                try:
                    data = json.loads(message["text"])
                except ValueError:
                    await ws.send_json({"type": "status", "text": "Dữ liệu gửi lên không hợp lệ."})
                    continue
                if data.get("type") == "chat":
                    await handle_question(ws, str(data.get("text", "")))
                else:
                    await ws.send_json({"type": "status", "text": "Loại tin nhắn chưa được hỗ trợ."})
            elif message.get("bytes") is not None:
                audio = message["bytes"]
                if not audio or len(audio) > MAX_AUDIO_BYTES:
                    await ws.send_json({"type": "status", "text": "Đoạn ghi âm trống hoặc quá dài."})
                    continue
                await ws.send_json({"type": "status", "text": "Bé Na đang nghe..."})
                try:
                    question = await transcribe_pcm(audio)
                    if not question:
                        await ws.send_json({"type": "status", "text": "Bé Na chưa nghe rõ. Bạn thử nói lại nhé."})
                        continue
                    await handle_question(ws, question)
                except Exception as exc:
                    logger.exception("Audio processing failed")
                    await ws.send_json({
                        "type": "status",
                        "text": f"Lỗi xử lý âm thanh: {str(exc)[:180]}",
                    })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("WebSocket error")
        try:
            await ws.close(code=1011)
        except Exception:
            pass
