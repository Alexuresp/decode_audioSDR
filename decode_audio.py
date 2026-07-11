"""
SDR-translate — захоплення аудіо з радіоефіру, STT та переклад українською.

Потоки:
  - Головний (main): захоплення через sounddevice, детекція мовлення Silero VAD,
    формування фрагментів, передача в audio_queue.
  - Обробки (process_audio_worker): Groq Whisper (транскрибація) + Llama (переклад),
    вивід оригіналу та перекладу в термінал.

Запуск:
    python decode_audio.py
Вихід: Ctrl+C. Потрібен GROQ_API_KEY у .env та аудіовхід (мікрофон/віртуальний кабель).

Адаптивний VAD (слабкий сигнал у шумах):
  - VAD_BASE_THRESHOLD  — жорстка нижня межа порогу.
  - VAD_NOISE_MARGIN    — запас порогу над шумовим полом (адаптується у тиші).
  - VAD_HYSTERESIS      — різниця старт/стоп, щоб не рвати фрази.
  - VAD_SMOOTH_ALPHA    — EMA-згладжування ймовірності.
  - VAD_NOISE_ALPHA     — швидкість адаптації шумового полу.
  - PREROLL_SECONDS     — буфер аудіо перед стартом (повертає обрізаний початок).
"""

import os
import wave
import numpy as np
import torch
import queue
import threading
import collections
import warnings
from dotenv import load_dotenv
from groq import Groq
import sounddevice as sd

# Приховуємо попередження від бібліотек
warnings.filterwarnings("ignore")
load_dotenv()

# ANSI escape-коди для виділення кольором у терміналі
COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_CYAN = '\033[96m'
COLOR_RESET = '\033[0m'

# ================= НАЛАШТУВАННЯ =================
# Аудіо
CHANNELS = 1
RATE = 16000
CHUNK = 512

# VAD (детекція голосу)
VAD_BASE_THRESHOLD = 0.5      # жорстка нижня межа порогу
VAD_NOISE_MARGIN  = 0.15      # запас над шумовим полом
VAD_HYSTERESIS    = 0.20      # різниця між порогом старту і стопу
VAD_SMOOTH_ALPHA  = 0.3       # згладжування ймовірності (0..1)
VAD_NOISE_ALPHA   = 0.005     # швидкість адаптації шумового полу
PREROLL_SECONDS   = 0.5       # буфер перед початком мовлення

SILENCE_DURATION = 0.2
SILENCE_CHUNKS = int(RATE / CHUNK * SILENCE_DURATION)
PREROLL_CHUNKS = int(RATE / CHUNK * PREROLL_SECONDS)

# Groq
GROQ_STT_MODEL = "whisper-large-v3"
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"

# Черга для передачі записаних фрагментів між потоками
audio_queue = queue.Queue()

# ================= ПОТІК ОБРОБКИ (STT + LLM) =================
def process_audio_worker():
    print("[Система] Ініціалізація Groq клієнта...")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("[Система] Groq готовий до роботи.")

    TEMP_AUDIO_FILE = "temp_audio_for_groq.wav"

    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            break

        try:
            audio_int16 = np.clip(audio_data * 32768, -32768, 32767).astype(np.int16)
            with wave.open(TEMP_AUDIO_FILE, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(RATE)
                wf.writeframes(audio_int16.tobytes())

            with open(TEMP_AUDIO_FILE, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(TEMP_AUDIO_FILE, file.read()),
                    model=GROQ_STT_MODEL,
                    response_format="json"
                )

            text = transcription.text.strip()

            if text:
                print(f"🎙️ [Оригінал]: {text}")

                prompt = (
                    "Ти професійний перекладач. Тема: радіоаматорство та побут. Переклади текст з радіоефіру українською мовою.\n\n"
                    "=== СЛОВНИК АБЕТОК ===\n"
                    "A: Alfa, Антон; B: Bravo, Богдан, Борис; C: Charlie, Центр, Цапля; D: Delta, Дмитро; "
                    "E: Echo, Еней, Елена; F: Foxtrot, Федір; G: Golf, Григорій, Галина; H: Hotel, Христина, Харитон; "
                    "I: India, Italy, Іван; J: Juliett, Йосип, йот, Иван краткий; K: Kilo, Кіловат, Константин; "
                    "L: Lima, Леонід; M: Mike, Марія, Михаил; N: November, Наталка, Николай; O: Oscar, Ольга; "
                    "P: Papa, Павло, Павел; Q: Quebec, Щука; R: Romeo, Radio, Роман; S: Sierra, Степан, Сергій; "
                    "T: Tango, Тетяна, Тарас, Тамара; U: Uniform, Україна, Ульяна; V: Victor, Жук, Женя; "
                    "W: Whiskey, Василь; X: X-ray, Ікс, Знак, Твёрдый знак; Y: Yankee, Ігрек; Z: Zulu, Зоя, Зинаида.\n\n"
                    "=== ІНСТРУКЦІЯ ЩОДО ПОЗИВНИХ ===\n"
                    "Якщо бачиш слова з цього словника, якими диктують HAM-позивний, переклади їх, "
                    "а після останнього слова позивного додай його латинську абревіатуру в дужках.\n"
                    "<example>\n"
                    "Вхід: Ульяна Роман пять Щука Женя\n"
                    "Вихід: Уляна Роман п'ять Щука Женя (UR5QV)\n"
                    "</example>\n\n"
                    "=== ОБМЕЖЕННЯ ===\n"
                    "Виведи ЛИШЕ переклад тексту, що знаходиться між тегами <text> і </text>. "
                    "Не коментуй, не пиши вступних слів і не дублюй приклад.\n\n"
                    f"<text>\n{text}\n</text>"
                )

                try:
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model=GROQ_LLM_MODEL,
                        temperature=0.3,
                    )
                    translation = chat_completion.choices[0].message.content.strip()
                    print(f"{COLOR_YELLOW}🇺🇦 [Переклад]: {translation}{COLOR_RESET}")
                except Exception as e:
                    print(f"⚠️ [Помилка Groq API]: {e}")
        except Exception as e:
            print(f"⚠️ [Помилка обробки аудіо]: {e}")
        finally:
            audio_queue.task_done()

# ================= ГОЛОВНИЙ ПОТІК (ЗАХОПЛЕННЯ ТА VAD) =================
def main():
    print("[Система] Завантаження Silero VAD...")
    vad_model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True
    )
    vad_model = vad_model.to('cpu')
    print("[Система] Silero VAD готовий.")

    worker = threading.Thread(target=process_audio_worker, daemon=True)
    worker.start()

    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(f"⚠️ [Аудіо статус]: {status}", flush=True)
        q.put(indata.copy())

    print("[Система] Слухаю ефір... (Натисніть Ctrl+C для виходу)")

    is_recording = False
    silence_counter = 0
    audio_buffer = []
    preroll_buffer = collections.deque(maxlen=PREROLL_CHUNKS)  # pre-roll
    noise_floor = 0.0                                          # шумовий пол
    speech_prob_smooth = 0.0                                   # згладжування

    def adaptive_threshold():
        # поріг = макс(базовий, шумовий пол + запас)
        return max(VAD_BASE_THRESHOLD, noise_floor + VAD_NOISE_MARGIN)

    try:
        with sd.InputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype='int16',
            blocksize=CHUNK,
            callback=callback
        ):
            while True:
                chunk = q.get()
                audio_np = chunk.astype(np.float32).flatten() / 32768.0
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

                speech_prob = vad_model(audio_tensor, RATE).item()

                # згладжування (скользяче середнє)
                speech_prob_smooth = (
                    (1 - VAD_SMOOTH_ALPHA) * speech_prob_smooth
                    + VAD_SMOOTH_ALPHA * speech_prob
                )

                # адаптація шумового полу тільки у тиші
                if not is_recording:
                    noise_floor = (
                        (1 - VAD_NOISE_ALPHA) * noise_floor
                        + VAD_NOISE_ALPHA * speech_prob
                    )

                start_th = adaptive_threshold()
                stop_th = start_th - VAD_HYSTERESIS              # гістерезис

                # pre-roll: завжди тримаємо останні чанки
                preroll_buffer.append(audio_np)

                if not is_recording:
                    if speech_prob_smooth > start_th:
                        is_recording = True
                        silence_counter = 0
                        print("▶", end="", flush=True)
                        # підклеюємо pre-roll до початку фрагмента
                        audio_buffer = list(preroll_buffer)
                else:
                    audio_buffer.append(audio_np)
                    if speech_prob_smooth < stop_th:
                        silence_counter += 1
                        if silence_counter > SILENCE_CHUNKS:
                            is_recording = False
                            print("■", flush=True)

                            full_audio = np.concatenate(audio_buffer)
                            audio_queue.put(full_audio.copy())

                            audio_buffer = []
                            silence_counter = 0
                    else:
                        silence_counter = 0

    except KeyboardInterrupt:
        print("\n[Система] Завершення роботи...")

if __name__ == "__main__":
    main()