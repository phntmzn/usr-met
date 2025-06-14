# === IMPORTS ===
import os
import subprocess
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from tqdm import tqdm
import numpy as np
from scipy.io import wavfile

from b import notes, chords, time_value_durations
from midiutil import MIDIFile

import objc
from Cocoa import NSObject
from Metal import *
from Metal import MTLCreateSystemDefaultDevice
from Foundation import NSData

# Helper for MTLSize creation (PyObjC doesn't provide MTLSizeMake)
def MTLSizeMake(width, height, depth):
    size = objc.createStructType('MTLSize', b'{MTLSize=QQQ}', ['width', 'height', 'depth'])
    return size(width, height, depth)

# === METAL SHADER COMPILATION AND LOADING ===
class MetalRenderer:
    def initMetal(self):
        self.device = MTLCreateSystemDefaultDevice()
        self.commandQueue = self.device.newCommandQueue()

        # Shader with multiple effects
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void audioPostProcess(device float* inAudio  [[ buffer(0) ]],
                                     device float* outAudio [[ buffer(1) ]],
                                     constant uint& effectType [[ buffer(2) ]],
                                     uint id [[ thread_position_in_grid ]]) {
            float input = inAudio[id];
            float output = 0.0;

            switch (effectType) {
                case 0:
                    output = input;
                    break;
                case 1:
                    output = input * 0.5;
                    break;
                case 2:
                    output = clamp(input * 5.0, -1.0, 1.0);
                    break;
                case 3:
                    output = (id % 100 < 90) ? input : 0.0;
                    break;
                case 4:
                    output = -input;
                    break;
                case 5:
                    output = inAudio[id % 512];
                    break;
                default:
                    output = input;
                    break;
            }
            outAudio[id] = output;
        }
        """

        with NamedTemporaryFile(delete=False, suffix=".metal") as metal_file:
            metal_file.write(shader_source.encode('utf-8'))
            metal_file_path = Path(metal_file.name)

        air_path = metal_file_path.with_suffix(".air")
        metallib_path = metal_file_path.with_suffix(".metallib")

        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", str(metal_file_path), "-o", str(air_path)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"[metal] Compilation failed:\n{result.stderr}")
        result = subprocess.run(
            ["xcrun", "metallib", str(air_path), "-o", str(metallib_path)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"[metallib] Linking failed:\n{result.stderr}")

        data = NSData.dataWithContentsOfFile_(str(metallib_path))
        self.library = self.device.newLibraryWithData_error_(data, None)
        self.kernel = self.library.newFunctionWithName_("audioPostProcess")
        self.pipeline = self.device.newComputePipelineStateWithFunction_error_(self.kernel, None)

    def processAudio(self, in_buffer, out_buffer, effect_type, sample_count):
        commandBuffer = self.commandQueue.commandBuffer()
        encoder = commandBuffer.computeCommandEncoder()

        encoder.setComputePipelineState_(self.pipeline)
        encoder.setBuffer_offset_atIndex_(in_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(out_buffer, 0, 1)

        # Add effect type as constant buffer
        effect_type_buf = self.device.newBufferWithLength_options_(4, 0)
        effect_type_ptr = objc.ObjCInstance(effect_type_buf.contents()).cast('I')
        effect_type_ptr[0] = effect_type
        encoder.setBuffer_offset_atIndex_(effect_type_buf, 0, 2)

        threads_per_threadgroup = MTLSizeMake(256, 1, 1)
        threadgroups = MTLSizeMake((sample_count + 255) // 256, 1, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroups, threads_per_threadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

# === CONFIGURATION ===
TOTAL_FILES = 200
TEMPO = 157
SAMPLE_RATE = 44100
SOUNDFONT_PATH = "/Users/macbookair/Downloads/OPLLandOPLL2DrumFix2Remake.sf2"
OUTPUT_DIR = Path.home() / "Desktop" / "wav_only_200"
FLUIDSYNTH_PATH = "/opt/homebrew/bin/fluidsynth"
POOL_SIZE = max(4, cpu_count())

# === CONSTANTS ===
BEATS_PER_MINUTE = TEMPO
DURATION_MINUTES = 2
TOTAL_BEATS = BEATS_PER_MINUTE * DURATION_MINUTES
BEATS_PER_BAR = 4
BARS = TOTAL_BEATS // BEATS_PER_BAR

# === LAYERED MIDI GENERATOR ===
def chord_pattern() -> str:
    duration = time_value_durations["sixteenth_note"]
    steps_per_beat = int(1 / duration)

    BEATS_PER_MINUTE = TEMPO
    DURATION_MINUTES = 2
    TOTAL_BEATS = BEATS_PER_MINUTE * DURATION_MINUTES
    BEATS_PER_BAR = 4
    BARS = TOTAL_BEATS // BEATS_PER_BAR
    total_steps = TOTAL_BEATS * steps_per_beat

    available_roots = list(notes.keys())
    available_chords = [c for c in chords.keys() if "7th" in c]

    random.seed()  # seed fresh each run
    root_cycle = random.choices(available_roots, k=8)
    chord_cycle = random.choices(available_chords, k=8)

    midi = MIDIFile(3)
    for t in range(3):
        midi.addTempo(t, 0, TEMPO)

    for step in range(total_steps):
        time = step * duration
        beat_index = step // steps_per_beat
        bar_index = beat_index // BEATS_PER_BAR
        beat_pos_in_bar = beat_index % BEATS_PER_BAR

        chord_index = bar_index % len(root_cycle)
        root = root_cycle[chord_index]
        chord_type = chord_cycle[chord_index]
        intervals = chords[chord_type]
        root_note = notes[root]

        if beat_pos_in_bar in [0, 2] and step % steps_per_beat == 0:
            bass_note = root_note + 12 * 3
            midi.addNote(0, 0, bass_note, time, 1.0, 100)

            for i in intervals[:3]:  # Triad
                chord_note = root_note + i + 12 * 5
                midi.addNote(1, 1, chord_note, time, 1.0, 100)

        if step % (steps_per_beat * 2) == 0:
            fx_note = root_note + (step % 4) + 12 * 6
            midi.addNote(2, 2, fx_note, time, duration, 110)

    tmp = NamedTemporaryFile(delete=False, suffix=".mid")
    with open(tmp.name, "wb") as f:
        midi.writeFile(f)

    return tmp.name


# === FLUIDSYNTH RENDER ===
def render_with_fluidsynth(midi_path, wav_path):
    subprocess.run([
        FLUIDSYNTH_PATH, "-ni", SOUNDFONT_PATH,
        midi_path, "-F", str(wav_path), "-r", str(SAMPLE_RATE)
    ], check=True)

# === METAL POST-EFFECT ===
def apply_metal_effect(wav_path: Path, effect_type: int):
    sr, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]  # Use only left channel

    norm_data = data.astype(np.float32) / 32768.0
    sample_count = len(norm_data)

    renderer = MetalRenderer()
    renderer.initMetal()

    in_buf = renderer.device.newBufferWithBytes_length_options_(
        norm_data.ctypes.data, norm_data.nbytes, 0)
    out_buf = renderer.device.newBufferWithLength_options_(
        norm_data.nbytes, 0)

    renderer.processAudio(in_buf, out_buf, effect_type, sample_count)

    out_ptr = objc.ObjCInstance(out_buf.contents()).cast('f')
    processed = np.frombuffer(out_ptr, dtype=np.float32, count=sample_count)
    int16_data = np.clip(processed * 32768.0, -32768, 32767).astype(np.int16)

    wavfile.write(wav_path, sr, int16_data)

# === FULL RENDER JOB ===
def render_one(index):
    wav_path = OUTPUT_DIR / f"{index:05}.wav"
    if wav_path.exists() and wav_path.stat().st_size > 4 * 1024 ** 3:
        print(f"⚠️ Skipping {wav_path.name}")
        return
    if wav_path.exists():
        wav_path.unlink()

    midi_path = chord_pattern()
    try:
        render_with_fluidsynth(midi_path, wav_path)
        effect = random.choice([1, 2, 3, 4, 5])
        apply_metal_effect(wav_path, effect)
        print(f"✅ Processed {wav_path.name} with effect {effect}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if os.path.exists(midi_path):
            os.unlink(midi_path)

# === MAIN ENTRY ===
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🎹 Generating {TOTAL_FILES} WAVs with GPU effects...")
    with ProcessPoolExecutor(max_workers=POOL_SIZE) as executor:
        futures = [executor.submit(render_one, i) for i in range(TOTAL_FILES)]
        for f in tqdm(as_completed(futures), total=TOTAL_FILES):
            try:
                f.result()
            except Exception as e:
                print(f"❌ Task failed: {e}")

if __name__ == "__main__":
    main()