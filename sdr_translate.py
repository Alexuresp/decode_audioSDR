#!/usr/bin/env python3
import os
import sys
import time
import wave
import queue
import argparse
import collections
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Force UTF-8 output on Windows to prevent console encoding issues ( characters)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Audio recording defaults
SAMPLE_RATE = 16000  # 16kHz is ideal for Whisper
CHANNELS = 1         # Mono audio
TEMP_FILENAME = "temp_SSB_audio.wav"

# Morse Code Latin and Cyrillic dictionaries
MORSE_LATIN = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '.-.-.-': '.', '--..--': ',', '..--..': '?', '.----.': "'", '-..-.': '/',
    '-...-': '=', '---...': ':', '-....-': '-', '.-.-.': '+',
}

MORSE_CYRILLIC = {
    '.-': 'А', '-...': 'Б', '.--': 'В', '--.': 'Г', '-..': 'Д',
    '.': 'Е', '...-': 'Ж', '--..': 'З', '..': 'И', '.---': 'Й',
    '-.-': 'К', '.-..': 'Л', '--': 'М', '-.': 'Н', '---': 'О',
    '.--.': 'П', '.-.': 'Р', '...': 'С', '-': 'Т', '..-': 'У',
    '..-.': 'Ф', '....': 'Х', '-.-.': 'Ц', '---.': 'Ч', '----': 'Ш',
    '--.-': 'Щ', '--.--': 'Ъ', '-.--': 'Ы', '-..-': 'Ь', '..-..': 'Э',
    '..--': 'Ю', '.-.-': 'Я',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '.-.-.-': '.', '--..--': ',', '..--..': '?', '.----.': "'", '-..-.': '/',
    '-...-': '=', '---...': ':', '-....-': '-', '.-.-.': '+',
}

def decode_morse_text(morse_str, cyrillic=False):
    """Decode a space-separated Morse string (e.g. '.- -...' or '.- / -...') into text."""
    dictionary = MORSE_CYRILLIC if cyrillic else MORSE_LATIN
    words = morse_str.strip().split(' / ')
    decoded_words = []
    
    for word in words:
        chars = word.split(' ')
        decoded_chars = [dictionary.get(ch, '?') for ch in chars if ch]
        decoded_words.append(''.join(decoded_chars))
        
    return ' '.join(decoded_words)

def decode_cw_audio(audio_data, samplerate):
    """
    Decodes Morse code from raw audio data using FFT and adaptive thresholding.
    Returns: (morse_symbols, latin_text, cyrillic_text)
    """
    # Convert input to float32
    audio = audio_data.astype(np.float32)
    # Normalize if int16
    if audio_data.dtype == np.int16:
        audio = audio / 32768.0
        
    # Flatten if multi-channel
    if len(audio.shape) > 1:
        audio = audio[:, 0]
        
    # 1. Tone Detection: find peak in 400Hz - 1000Hz CW range
    n = len(audio)
    if n < 1024:
        return "", "", ""
        
    frequencies = np.fft.rfftfreq(n, d=1/samplerate)
    fft_magnitudes = np.abs(np.fft.rfft(audio))
    
    # Mask for CW band
    mask = (frequencies >= 400.0) & (frequencies <= 1000.0)
    if not np.any(mask):
        return "", "", ""
        
    peak_idx = np.argmax(fft_magnitudes[mask])
    target_freq = frequencies[mask][peak_idx]
    
    # 2. Envelope tracking using Short-Time Fourier Transform (STFT)
    window_size = 512
    hop_size = 128
    time_step = hop_size / samplerate
    
    n_frames = (len(audio) - window_size) // hop_size + 1
    if n_frames < 10:
        return "", "", ""
        
    bin_idx = int(round(target_freq * window_size / samplerate))
    win = np.hanning(window_size)
    
    envelope = []
    for i in range(n_frames):
        start = i * hop_size
        end = start + window_size
        chunk = audio[start:end] * win
        fft_data = np.fft.rfft(chunk)
        
        # Take max around target bin to allow slight frequency drift
        start_bin = max(0, bin_idx - 1)
        end_bin = min(len(fft_data), bin_idx + 2)
        val = np.max(np.abs(fft_data[start_bin:end_bin]))
        envelope.append(val)
        
    envelope = np.array(envelope)
    
    # 3. Adaptive Thresholding
    min_val = np.percentile(envelope, 15)
    max_val = np.percentile(envelope, 95)
    
    # Check signal-to-noise ratio
    if max_val - min_val < 1e-4 or max_val / (min_val + 1e-6) < 1.8:
        return "", "", ""
        
    threshold = min_val + 0.4 * (max_val - min_val)
    binary = (envelope > threshold).astype(int)
    
    # 4. Run-length encoding
    runs = []
    current_val = binary[0]
    current_len = 1
    for val in binary[1:]:
        if val == current_val:
            current_len += 1
        else:
            runs.append((current_val, current_len))
            current_val = val
            current_len = 1
    runs.append((current_val, current_len))
    
    # Filter short noise glitches (less than 15ms)
    min_frames = int(0.015 / time_step)
    filtered_runs = []
    for val, length in runs:
        if length >= min_frames or val == 0:
            filtered_runs.append((val, length))
            
    # 5. Cluster lengths of On periods (dots vs dashes) and Off periods (spaces)
    on_lengths = [length for val, length in filtered_runs if val == 1]
    if not on_lengths:
        return "", "", ""
        
    sorted_on = sorted(on_lengths)
    # Estimate dot_len (20th percentile is a good robust estimate for shortest key-down length)
    dot_len = max(1.0, np.percentile(sorted_on, 20))
    dash_len = dot_len * 3
    on_threshold = (dot_len + dash_len) / 2.0
    
    # Space thresholds
    char_space_threshold = dot_len * 2.0
    word_space_threshold = dot_len * 5.0
    
    # Decode to morse codes
    morse_symbols = []
    for val, length in filtered_runs:
        if val == 1:
            if length > on_threshold:
                morse_symbols.append("-")
            else:
                morse_symbols.append(".")
        else: # val == 0
            if length > word_space_threshold:
                morse_symbols.append(" / ")
            elif length > char_space_threshold:
                morse_symbols.append(" ")
                
    morse_str = "".join(morse_symbols).strip()
    if not morse_str:
        return "", "", ""
        
    # Decode to text
    latin_text = decode_morse_text(morse_str, cyrillic=False)
    cyrillic_text = decode_morse_text(morse_str, cyrillic=True)
    
    # Format target frequency for display
    print(f"\r[+] CW Tone Detected at {target_freq:.1f} Hz (Est. dot length: {dot_len * time_step * 1000:.0f} ms)")
    
    return morse_str, latin_text, cyrillic_text

def get_groq_client():
    """Initialize and return the Groq client, prompting for the API key if missing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n[!] GROQ_API_KEY environment variable not found.")
        api_key = input("Please enter your Groq API Key: ").strip()
        if not api_key:
            print("Error: Groq API Key is required to run this script.")
            sys.exit(1)
        # Save to .env file for convenience next time
        with open(".env", "a") as f:
            f.write(f"\nGROQ_API_KEY={api_key}\n")
        print("API Key saved to .env file.")
        # Reload environment
        load_dotenv()
    
    return Groq(api_key=api_key)

def list_and_select_device():
    """List input devices and attempt to auto-select an SDR/USB interface, or ask the user.
    Remembers the last choice in .env (SDR_DEVICE_INDEX) for instant restart."""
    devices = sd.query_devices()
    input_devices = []

    print("\n=== Available Audio Input Devices ===")
    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append((idx, dev['name']))
            print(f" [{len(input_devices) - 1}] Index {idx}: {dev['name']} (Channels: {dev['max_input_channels']}, Rate: {dev['default_samplerate']}Hz)")

    if not input_devices:
        print("\nError: No audio input devices found.")
        sys.exit(1)

    # --- Check for remembered device ---
    saved_global_idx = os.getenv("SDR_DEVICE_INDEX")
    if saved_global_idx is not None:
        try:
            saved_global_idx = int(saved_global_idx)
            # Verify it still exists
            match = [name for gi, name in input_devices if gi == saved_global_idx]
            if match:
                print(f"\n[+] Using remembered device: {match[0]} (Index {saved_global_idx})")
                print("    (Run with --device to override, or delete SDR_DEVICE_INDEX from .env to re-select)")
                return saved_global_idx
        except (ValueError, TypeError):
            pass

    # --- Auto-detection logic ---
    matched_devices = []
    for local_idx, (global_idx, name) in enumerate(input_devices):
        name_lower = name.lower()
        if "sdr" in name_lower or "codec" in name_lower or "usb" in name_lower:
            matched_devices.append((local_idx, global_idx, name))

    def save_and_return(global_idx):
        """Persist selection to .env and return the index."""
        env_path = ".env"
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        # Remove any existing SDR_DEVICE_INDEX line
        lines = [l for l in lines if not l.strip().startswith("SDR_DEVICE_INDEX=")]
        lines.append(f"SDR_DEVICE_INDEX={global_idx}\n")
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"[+] Device index {global_idx} saved to .env — will be used automatically next time.")
        return global_idx

    if len(matched_devices) == 1:
        local_idx, global_idx, name = matched_devices[0]
        print(f"\n[+] Auto-selected device: {name} (System Index {global_idx})")
        return save_and_return(global_idx)
    elif len(matched_devices) > 1:
        print("\n[+] Multiple potential SDR/USB audio devices detected:")
        for local_idx, global_idx, name in matched_devices:
            print(f"  Option [{local_idx}]: {name}")

        while True:
            try:
                choice = input(f"Select option number [0-{len(input_devices)-1}] (Press Enter for default {matched_devices[0][2]}): ").strip()
                if not choice:
                    return save_and_return(matched_devices[0][1])
                choice_idx = int(choice)
                if 0 <= choice_idx < len(input_devices):
                    return save_and_return(input_devices[choice_idx][0])
            except ValueError:
                pass
            print("Invalid selection. Please enter a valid number.")

    # Fallback to prompt
    print("\nCould not automatically narrow down to a single SDR/USB device.")
    while True:
        try:
            choice = input(f"Please select the device index [0-{len(input_devices)-1}] to use: ").strip()
            choice_idx = int(choice)
            if 0 <= choice_idx < len(input_devices):
                return save_and_return(input_devices[choice_idx][0])
        except ValueError:
            pass
        print("Invalid selection. Please enter a valid number.")

WHISPER_RATE = 16000  # Whisper works best at 16 kHz

def downsample_for_whisper(data, device_rate):
    """
    Downsample int16 audio from device_rate to WHISPER_RATE (16000 Hz).
    Uses simple decimation when the ratio is an integer (e.g. 48000/16000=3),
    otherwise falls back to linear interpolation — no scipy required.
    """
    if device_rate == WHISPER_RATE:
        return data, WHISPER_RATE

    # Flatten to mono if needed
    audio = data[:, 0] if len(data.shape) > 1 else data
    audio = audio.astype(np.float32)

    ratio = device_rate / WHISPER_RATE
    if ratio == int(ratio):
        # Fast path: integer decimation (48000→16000 = take every 3rd sample)
        step = int(ratio)
        resampled = audio[::step]
    else:
        # General path: linear interpolation
        original_len = len(audio)
        new_len = int(original_len / ratio)
        old_t = np.linspace(0, 1, original_len)
        new_t = np.linspace(0, 1, new_len)
        resampled = np.interp(new_t, old_t, audio)

    return resampled.astype(np.int16), WHISPER_RATE

def save_wav(filename, data, sample_rate):
    """Save raw audio data to a WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit is 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())

def transcribe_and_translate(client, filename, whisper_lang, target_lang, translation_model):
    """Send audio to Groq Whisper and translate the output using Llama."""
    print(" [Transcribing with Whisper...]", end="", flush=True)
    
    try:
        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3",
                response_format="json",
                language=whisper_lang if whisper_lang else None
            )
        
        text = transcription.text.strip()
        if not text:
            print("\r -> [No speech detected]")
            return

        print(f"\r[Original]: {text}")
        
        # Translation Step
        print(" [Translating...]", end="", flush=True)
        
        prompt = f"""You are an expert translator specializing in radio communications, military, and amateur (SDR/SSB) radio conversations.
Translate the following transcribed text from the radio to {target_lang}.
- Preserve call signs, signal reports (like 59, 599), and standard radio abbreviations (Q-codes, Roger, Wilco, etc.).
- Maintain the original tone and formatting (e.g. static noises, incomplete sentences).
- If the text is purely noise or contains no meaningful words, output "[No clear voice detected]".
- Provide ONLY the translation, nothing else. No introductions, headers, or explanations.

Text to translate:
"{text}"
"""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=translation_model,
            temperature=0.3,
        )
        
        translation = chat_completion.choices[0].message.content.strip()
        print(f"\r[{target_lang}]: {translation}\n" + "-"*50)
        
    except Exception as e:
        print(f"\rError during API call: {e}")

def process_audio(client, filename, whisper_lang, target_lang, translation_model, signal_type, samplerate, audio_data=None):
    """Process the captured audio based on whether it is voice or CW."""
    if signal_type == "cw":
        # Decode Morse locally
        if audio_data is None:
            # Load from WAV if data not passed directly
            try:
                with wave.open(filename, 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio_data = np.frombuffer(frames, dtype=np.int16)
            except Exception as e:
                print(f"Error reading audio file: {e}")
                return
                
        morse_symbols, latin, cyrillic = decode_cw_audio(audio_data, samplerate)
        if not morse_symbols:
            print("\r -> [No Morse code detected]")
            return
            
        print(f"\r[Morse]: {morse_symbols}")
        print(f"[Latin]: {latin}")
        print(f"[Cyrillic]: {cyrillic}")
        
        # Translate / Expand abbreviations using Groq LLM
        print(" [Expanding CW shorthand...]", end="", flush=True)
        try:
            prompt = f"""You are an expert amateur radio (ham) operator and translator.
Translate and expand the following Morse code transmission (which contains standard ham radio shorthand like CQ, DE, 73, UR, RST 599, call signs, etc.) into natural, readable {target_lang}.
- Latin text: "{latin}"
- Cyrillic equivalent: "{cyrillic}"
- Expand abbreviations: CQ (general call to all stations), DE (from), 73 (best regards), UR (your/you are), RST 599 (excellent signal report), K (over/go ahead), etc.
- Identify the callsigns present in the transmission.
- If the text looks like random characters or noise, output "[Incomplete or noisy Morse code]".
- Provide ONLY the translation/explanation, nothing else. No introductions, headers, or explanations.

Expanded Translation:
"""
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=translation_model,
                temperature=0.3,
            )
            translation = chat_completion.choices[0].message.content.strip()
            print(f"\r[{target_lang}]: {translation}\n" + "-"*50)
        except Exception as e:
            print(f"\rError translating Morse: {e}")
    else:
        # Standard voice translation (Whisper + LLM)
        transcribe_and_translate(client, filename, whisper_lang, target_lang, translation_model)

def run_vad_voice_mode(client, device_idx, samplerate, whisper_lang, target_lang, translation_model,
                       vad_threshold=0.5, silence_duration=0.2):
    """
    Continuous VAD-driven voice recorder for SSB.

    Listens in small chunks, detects voice activity via RMS energy,
    captures the phrase (with pre-roll), then sends to Whisper + LLM
    only when real speech was present — no fixed-length blocks needed.

    Parameters:
      vad_threshold   – normalized RMS threshold (0.0–1.0); lower = more sensitive
      silence_duration – seconds of silence after speech before phrase ends
    """
    CHUNK        = 1024                               # ~21 ms @ 48kHz
    time_per_chunk = CHUNK / samplerate               # seconds per chunk
    silence_chunks = int(silence_duration / time_per_chunk)
    preroll_chunks = int(0.3 / time_per_chunk)        # 300 ms preroll — catches word starts

    print(f"\n[*] VAD Voice Mode | Device rate: {samplerate} Hz → Whisper: {WHISPER_RATE} Hz")
    print(f"[*] VAD threshold: {vad_threshold:.2f} | Silence cutoff: {silence_duration:.1f}s ({silence_chunks} chunks)")
    print("[*] Listening... Press Ctrl+C to stop.")
    print("-" * 50)

    q = queue.Queue()
    def callback(indata, frames, ts, status):
        q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=samplerate, channels=1, dtype='int16',
        device=device_idx, blocksize=CHUNK, callback=callback
    )

    def rms(chunk):
        """Normalized RMS energy: 0.0 (silence) → 1.0 (full scale)."""
        return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) / 32768.0

    preroll   = collections.deque(maxlen=preroll_chunks)  # ring buffer before speech
    recording = []
    in_speech  = False
    silent_cnt = 0

    with stream:
        try:
            while True:
                try:
                    chunk = q.get(timeout=1.0)[:, 0]
                except queue.Empty:
                    continue

                energy = rms(chunk)

                if not in_speech:
                    preroll.append(chunk.copy())
                    if energy > vad_threshold:
                        in_speech  = True
                        silent_cnt = 0
                        recording  = list(preroll)   # include preroll so first word isn't clipped
                        print("▶ ", end="", flush=True)
                else:
                    recording.append(chunk.copy())
                    if energy <= vad_threshold:
                        silent_cnt += 1
                        if silent_cnt >= silence_chunks:
                            # ── End of phrase ──────────────────────────────
                            print("■", flush=True)
                            in_speech  = False
                            silent_cnt = 0
                            preroll.clear()

                            audio = np.concatenate(recording)
                            recording = []

                            # Minimum phrase length check (avoid tiny glitches)
                            if len(audio) < samplerate * 0.5:
                                print("  (too short, skipping)")
                                continue

                            # Downsample → 16 kHz → WAV → Whisper → translate
                            wav_data, wav_rate = downsample_for_whisper(audio, samplerate)
                            save_wav(TEMP_FILENAME, wav_data, wav_rate)
                            transcribe_and_translate(client, TEMP_FILENAME,
                                                     whisper_lang, target_lang, translation_model)
                    else:
                        silent_cnt = 0   # reset silence counter while speech continues

        except KeyboardInterrupt:
            print("\n[*] VAD stopped.")
        finally:
            if os.path.exists(TEMP_FILENAME):
                os.remove(TEMP_FILENAME)

def run_continuous_mode(client, device_idx, duration, whisper_lang, target_lang, translation_model, samplerate, signal_type):
    """Record audio in continuous blocks and process them sequentially."""
    print(f"\n[*] Starting Continuous Loop Mode. Recording in {duration} second chunks...")
    print(f"[*] Signal type: {signal_type.upper()}")
    print(f"[*] Using sample rate: {samplerate}Hz")
    print("[*] Press Ctrl+C to stop.")
    print("-" * 50)
    
    try:
        while True:
            print(f"Recording {duration}s block...", end="", flush=True)
            
            # Record block
            recording = sd.rec(
                int(duration * samplerate), 
                samplerate=samplerate, 
                channels=CHANNELS, 
                dtype='int16', 
                device=device_idx
            )
            sd.wait()
            print(" done. Processing...")
            
            # Downsample to 16kHz for Whisper (e.g. 48000→16000, factor 3)
            wav_data, wav_rate = downsample_for_whisper(recording, samplerate)
            save_wav(TEMP_FILENAME, wav_data, wav_rate)

            # Process block (pass native samplerate for CW, wav_rate used inside WAV file)
            process_audio(client, TEMP_FILENAME, whisper_lang, target_lang, translation_model, signal_type, samplerate, recording)
            
    except KeyboardInterrupt:
        print("\n\n[*] Stopped Continuous Mode.")
    finally:
        # Cleanup
        if os.path.exists(TEMP_FILENAME):
            os.remove(TEMP_FILENAME)

def run_manual_mode(client, device_idx, whisper_lang, target_lang, translation_model, samplerate, signal_type):
    """Record audio when user presses Enter, stop when they press Enter again."""
    print("\n[*] Starting Manual Mode.")
    print(f"[*] Signal type: {signal_type.upper()}")
    print(f"[*] Using sample rate: {samplerate}Hz")
    print("[*] Press Enter to start recording, and Enter again to stop.")
    print("[*] Press Ctrl+C to exit.")
    print("-" * 50)
    
    try:
        while True:
            input(">> Press [Enter] to START recording...")
            
            q = queue.Queue()
            
            def callback(indata, frames, time, status):
                if status:
                    print(status, file=sys.stderr)
                q.put(indata.copy())
            
            # Start stream
            stream = sd.InputStream(
                samplerate=samplerate, 
                channels=CHANNELS, 
                dtype='int16', 
                device=device_idx,
                callback=callback
            )
            
            print("Recording... (Press [Enter] to STOP and process)")
            with stream:
                input()
            
            print("Processing audio...")
            
            # Collect data from queue
            audio_data = []
            while not q.empty():
                audio_data.append(q.get())
            
            if not audio_data:
                print("No audio captured.")
                continue
                
            recording = np.concatenate(audio_data, axis=0)
            
            # Downsample to 16kHz for Whisper (e.g. 48000→16000, factor 3)
            wav_data, wav_rate = downsample_for_whisper(recording, samplerate)
            save_wav(TEMP_FILENAME, wav_data, wav_rate)

            # Process block (pass native samplerate for CW, wav_rate used inside WAV file)
            process_audio(client, TEMP_FILENAME, whisper_lang, target_lang, translation_model, signal_type, samplerate, recording)
            
    except (KeyboardInterrupt, EOFError):
        print("\n\n[*] Stopped Manual Mode.")
    finally:
        # Cleanup
        if os.path.exists(TEMP_FILENAME):
            os.remove(TEMP_FILENAME)

def run_realtime_cw(device_idx, samplerate, morse_lang="latin", manual_freq=None):
    """Real-time Morse code decoder — prints decoded characters directly, no LLM involved."""
    dictionary = MORSE_CYRILLIC if morse_lang == "cyrillic" else MORSE_LATIN

    if manual_freq is not None:
        target_freq = manual_freq
        print(f"\n[+] Locked onto manual CW Tone: {target_freq:.1f} Hz")
    else:
        print("\n[*] Starting Real-time Morse (CW) Decoder.")
        print("[*] Calibrating tone frequency... (1.5 seconds — tune to an active CW signal)...")

        cal_data = sd.rec(int(1.5 * samplerate), samplerate=samplerate, channels=1, dtype='float32', device=device_idx)
        sd.wait()

        freqs = np.fft.rfftfreq(len(cal_data), d=1/samplerate)
        mags  = np.abs(np.fft.rfft(cal_data[:, 0]))
        mask  = (freqs >= 400.0) & (freqs <= 1000.0)

        if np.any(mask):
            target_freq = freqs[mask][np.argmax(mags[mask])]
            print(f"[+] Calibrated! Locked onto CW Tone: {target_freq:.1f} Hz")
        else:
            target_freq = 700.0
            print(f"[!] No strong tone found, defaulting to {target_freq:.0f} Hz")

    print("[*] Decoding live. Press Ctrl+C to exit.")
    print("-" * 50)

    block_size = 512
    bin_idx = int(round(target_freq * block_size / samplerate))

    q = queue.Queue()
    def callback(indata, frames, ts, status):
        q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=samplerate, channels=1, dtype='float32',
        device=device_idx, callback=callback, blocksize=block_size
    )

    dot_len_ms      = 80.0                          # adaptive dot length estimate
    env_buf         = collections.deque(maxlen=300) # ~3 s rolling window for threshold
    current_state   = 0                             # 0 = key up, 1 = key down
    state_dur       = 0
    char_buf        = []
    space_printed   = False

    with stream:
        try:
            while True:
                try:
                    chunk = q.get_nowait()[:, 0]
                except queue.Empty:
                    time.sleep(0.002)
                    continue

                # --- Envelope magnitude at target bin ---
                win     = np.hanning(len(chunk))
                fft_out = np.fft.rfft(chunk * win)
                sb      = max(0, bin_idx - 1)
                eb      = min(len(fft_out), bin_idx + 2)
                mag     = float(np.max(np.abs(fft_out[sb:eb])))
                env_buf.append(mag)

                # --- Adaptive threshold ---
                if len(env_buf) >= 50:
                    lo = np.percentile(env_buf, 15)
                    hi = np.percentile(env_buf, 90)
                else:
                    lo, hi = 0.001, 0.05

                if hi - lo < 0.002 or hi / (lo + 1e-6) < 1.6:
                    is_down = 0
                else:
                    is_down = 1 if mag > lo + 0.35 * (hi - lo) else 0

                # --- State machine ---
                if is_down == current_state:
                    state_dur += 1
                else:
                    dur_ms = state_dur * (block_size / samplerate) * 1000.0

                    if current_state == 1 and dur_ms >= 12.0:   # key just released
                        if dur_ms > dot_len_ms * 2.0:
                            char_buf.append("-")
                            dot_len_ms = 0.9*dot_len_ms + 0.1*(dur_ms/3.0)
                        else:
                            char_buf.append(".")
                            dot_len_ms = 0.9*dot_len_ms + 0.1*dur_ms
                        dot_len_ms    = max(30.0, min(dot_len_ms, 240.0))
                        space_printed = False

                    current_state = is_down
                    state_dur     = 1

                # --- Emit character / word space ---
                if current_state == 0:
                    idle_ms = state_dur * (block_size / samplerate) * 1000.0

                    if idle_ms > dot_len_ms * 2.2 and char_buf:
                        ch = dictionary.get("".join(char_buf), "?")
                        print(ch, end="", flush=True)
                        char_buf = []

                    if idle_ms > dot_len_ms * 5.5 and not space_printed:
                        print(" ", end="", flush=True)
                        space_printed = True

                    # Blank line after long pause (end of transmission)
                    if idle_ms > dot_len_ms * 20 and not space_printed:
                        print()
                        space_printed = True

        except KeyboardInterrupt:
            print("\n[*] Stopped.")

def main():
    parser = argparse.ArgumentParser(description="SSB Transceiver Live Voice Decoder and Translator using Groq API")
    parser.add_argument("--mode", choices=["continuous", "manual"], default="continuous",
                        help="Continuous block recording (default) or manual start/stop")
    parser.add_argument("--type", choices=["voice", "cw"], default="voice",
                        help="Type of signal to decode: 'voice' (SSB speech) or 'cw' (Morse code)")
    parser.add_argument("--duration", type=int, default=10,
                        help="Duration of recording block in continuous mode (default: 10s)")
    parser.add_argument("--target-lang", type=str, default=None,
                        help="Target translation language (defaults to environment or 'Ukrainian')")
    parser.add_argument("--whisper-lang", type=str, default=None,
                        help="Hint language for Whisper to improve accuracy (e.g., 'en', 'uk')")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile",
                        help="Groq LLM model for translation (default: llama-3.3-70b-versatile)")
    parser.add_argument("--device", type=int, default=None,
                        help="Device index of sound card (bypasses interactive selection)")
    parser.add_argument("--cw-lang", choices=["latin", "cyrillic"], default="latin",
                        help="Morse decode language (default: latin)")
    parser.add_argument("--cw-freq", type=float, default=None,
                        help="Manual CW tone frequency in Hz (bypasses auto-calibration)")
    args = parser.parse_args()

    # Determine default target language
    target_lang = args.target_lang or os.getenv("DEFAULT_TARGET_LANGUAGE", "Ukrainian")
    whisper_lang = args.whisper_lang or os.getenv("WHISPER_LANGUAGE", "")
    
    # Initialize Groq Client
    client = get_groq_client()
    
    # Select audio device
    if args.device is not None:
        device_idx = args.device
        print(f"Using device index: {device_idx}")
    else:
        device_idx = list_and_select_device()
        
    # Get the default sample rate of the selected device to prevent PortAudio errors
    try:
        device_info = sd.query_devices(device_idx, 'input')
        samplerate = int(device_info.get('default_samplerate', 16000))
        print(f"[+] Recording sample rate set to: {samplerate}Hz")
    except Exception as e:
        print(f"[!] Warning: could not query device info ({e}). Defaulting to 16000Hz.")
        samplerate = 16000

    if args.type == "cw":
        run_realtime_cw(device_idx, samplerate, args.cw_lang, args.cw_freq)
    elif args.mode == "continuous":
        run_continuous_mode(client, device_idx, args.duration, whisper_lang, target_lang, args.model, samplerate, args.type)
    else:
        run_manual_mode(client, device_idx, whisper_lang, target_lang, args.model, samplerate, args.type)

if __name__ == "__main__":
    main()
