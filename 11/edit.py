# === IMPORTS ===
import os
import subprocess
import random
import shutil
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import ctypes
from tqdm import tqdm
from scipy.io import wavfile

from b import notes, chords, time_value_durations  # Custom music definitions
from midiutil import MIDIFile  # MIDI file writer

# macOS system libraries and Metal framework via PyObjC
import objc
from Cocoa import NSObject
from Metal import *
from Metal import MTLCreateSystemDefaultDevice
from Foundation import NSData
import plistlib
import uuid
import tempfile

# Import MTLCreateSystemDefaultDevice from Metal
from objc import lookUpClass

# Helper for MTLSizeMake (since PyObjC does not provide it directly)
import collections
MTLSize = collections.namedtuple("MTLSize", ["width", "height", "depth"])
def MTLSizeMake(width, height, depth):
    return MTLSize(width, height, depth)

# === LAUNCH AGENT INJECTOR ===
class PlistDylibInjector:
    """
    This class builds a .dylib, creates a launch agent plist that loads it with DYLD_INSERT_LIBRARIES,
    and installs it into ~/Library/LaunchAgents.
    """
    def __init__(self, name="com.apple.fake", label_suffix=None, backup_dir=Path.home() / "LaunchAgentBackups"):
        self.name = name
        self.label = f"{self.name}.{label_suffix or uuid.uuid4().hex[:6]}"
        self.temp_dir = Path(tempfile.mkdtemp())  # Temporary directory
        self.dylib_path = self.temp_dir / f"{self.label}.dylib"
        self.plist_path = self.temp_dir / f"{self.label}.plist"
        self.agent_path = Path.home() / "Library" / "LaunchAgents" / f"{self.label}.plist"
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def build_dylib_from_stub(self, dylib_name=None):
        """
        Writes and compiles a simple Objective-C dynamic library stub that logs a message when loaded.
        """
        stub_code = """
        #include <stdio.h>
        #include <objc/runtime.h>
        #include <objc/message.h>
        #import <Foundation/Foundation.h>

        __attribute__((constructor))
        static void load() {
            NSLog(@"[DYLIB] Injected successfully.");
            FILE *f = fopen("/tmp/injection_log.txt", "a+");
            if (f) { fputs("Injected!\\n", f); fclose(f); }
        }
        """
        c_path = self.temp_dir / "stub.m"
        out_dylib = self.dylib_path if not dylib_name else Path(dylib_name)

        with open(c_path, "w") as f:
            f.write(stub_code)

        compile_cmd = [
            "clang", "-dynamiclib", str(c_path),
            "-o", str(out_dylib),
            "-framework", "Foundation"
        ]
        subprocess.run(compile_cmd, check=True)
        os.chmod(out_dylib, 0o755)
        return out_dylib

    def write_plist(self, program_args=None, run_at_load=True, keep_alive=True, interval=60):
        """
        Writes a launch agent plist to insert the dylib into a specified executable.
        """
        if not program_args:
            program_args = ["/usr/bin/true"]

        plist_data = {
            "Label": self.label,
            "ProgramArguments": program_args,
            "RunAtLoad": run_at_load,
            "EnvironmentVariables": {
                "DYLD_INSERT_LIBRARIES": str(self.dylib_path)
            },
            "KeepAlive": keep_alive,
            "StartInterval": interval
        }

        with open(self.plist_path, "wb") as f:
            plistlib.dump(plist_data, f)

    def install_to_launch_agents(self):
        """
        Installs the plist to ~/Library/LaunchAgents and backs up any existing file.
        """
        if self.agent_path.exists():
            timestamp = int(time.time())
            backup_path = self.backup_dir / f"{self.label}.{timestamp}.bak.plist"
            shutil.copy(self.agent_path, backup_path)
            print(f"[🔄] Backed up existing plist to: {backup_path}")

        shutil.copy(self.plist_path, self.agent_path)
        print(f"[🧩] Installed plist to LaunchAgents: {self.agent_path}")

    def load_plist(self):
        subprocess.run(["launchctl", "load", str(self.agent_path)], check=True)

    def unload_plist(self):
        if self.agent_path.exists():
            subprocess.run(["launchctl", "unload", str(self.agent_path)], check=True)

    def cleanup(self):
        """
        Removes all temporary files and the LaunchAgent plist.
        """
        self.unload_plist()
        for path in [self.dylib_path, self.plist_path]:
            if path.exists():
                path.unlink()
        for file in self.temp_dir.iterdir():
            if file.exists():
                file.unlink()
        if self.temp_dir.exists():
            self.temp_dir.rmdir()

    def info(self):
        """
        Returns metadata about the current dylib and plist setup.
        """
        return {
            "plist": str(self.plist_path),
            "dylib": str(self.dylib_path),
            "label": self.label,
            "installed": str(self.agent_path),
            "temp_dir": str(self.temp_dir)
        }

# === METAL STRUCTS ===
class EffectParams(ctypes.Structure):
    """
    Structure for passing GPU audio effect parameters to Metal kernel.
    """
    _fields_ = [
        ("effectType", ctypes.c_uint),
        ("dryWetMix", ctypes.c_float),
        ("gain", ctypes.c_float),
    ]

# === METAL RENDERER ===
class MetalRenderer:
    """
    Compiles and runs a Metal compute shader for real-time audio processing.
    """
    def initMetal(self):
        self.device = MTLCreateSystemDefaultDevice()
        if self.device is None:
            # Try to get the function from the global namespace if not imported
            try:
                self.device = objc.lookUpClass("MTLCreateSystemDefaultDevice")()
            except Exception:
                raise RuntimeError("Could not create Metal device: MTLCreateSystemDefaultDevice is not available.")
        self.commandQueue = self.device.newCommandQueue()

        # === Metal shader kernel ===
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        struct EffectParams {
            uint effectType;
            float dryWetMix;
            float gain;
        };

        kernel void audioPostProcess(device float* inAudio  [[ buffer(0) ]],
                                     device float* outAudio [[ buffer(1) ]],
                                     constant EffectParams& params [[ buffer(2) ]],
                                     uint id [[ thread_position_in_grid ]]) {
            float input = inAudio[id];
            float output = 0.0;

            switch (params.effectType) {
                case 0: output = input; break;
                case 1: output = input * params.gain; break;
                case 2: output = clamp(input * 5.0, -1.0, 1.0); break;
                case 3: output = (id % 100 < 90) ? input : 0.0; break;
                case 4: output = -input; break;
                case 5: output = inAudio[id % 512]; break;
                default: output = input; break;
            }

            outAudio[id] = (1.0 - params.dryWetMix) * input + params.dryWetMix * output;
        }
        """

        # Write and compile the Metal shader
        with NamedTemporaryFile(delete=False, suffix=".metal") as metal_file:
            metal_file.write(shader_source.encode("utf-8"))
            metal_file_path = Path(metal_file.name)

        air_path = metal_file_path.with_suffix(".air")
        metallib_path = metal_file_path.with_suffix(".metallib")

        # Compile to .air and then .metallib
        subprocess.run(["xcrun", "-sdk", "macosx", "metal", str(metal_file_path), "-o", str(air_path)], check=True)
        subprocess.run(["xcrun", "metallib", str(air_path), "-o", str(metallib_path)], check=True)

        data = NSData.dataWithContentsOfFile_(str(metallib_path))
        self.library = self.device.newLibraryWithData_error_(data, None)
        self.kernel = self.library.newFunctionWithName_("audioPostProcess")
        self.pipeline = self.device.newComputePipelineStateWithFunction_error_(self.kernel, None)

    def processAudio(self, in_buffer, out_buffer, param_buffer, sample_count):
        """
        Dispatches the Metal compute shader to process audio in parallel on GPU.
        """
        commandBuffer = self.commandQueue.commandBuffer()
        encoder = commandBuffer.computeCommandEncoder()

        encoder.setComputePipelineState_(self.pipeline)
        encoder.setBuffer_offset_atIndex_(in_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(out_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(param_buffer, 0, 2)

        threads_per_threadgroup = MTLSizeMake(256, 1, 1)
        threadgroups = MTLSizeMake((sample_count + 255) // 256, 1, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadgroups, threads_per_threadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

# === AUDIO EFFECT ENGINE ===
class AudioEffectEngine:
    """
    Engine that handles numpy <-> Metal buffer conversion and audio processing pipeline.
    """
    def __init__(self):
        self.renderer = MetalRenderer.alloc().init()
        self.renderer.initMetal()

    def apply(self, np_data: np.ndarray, params: EffectParams) -> np.ndarray:
        sample_count = len(np_data)
        norm_data = np_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

        in_buf = self.renderer.device.newBufferWithBytes_length_options_(
            norm_data.ctypes.data, norm_data.nbytes, 0)
        out_buf = self.renderer.device.newBufferWithLength_options_(
            norm_data.nbytes, 0)
        param_buf = self.renderer.device.newBufferWithBytes_length_options_(
            ctypes.byref(params), ctypes.sizeof(params), 0)

        self.renderer.processAudio(in_buf, out_buf, param_buf, sample_count)

        # Convert Metal buffer to numpy array
        processed = np.frombuffer(out_buf.contents().bytes(), dtype=np.float32, count=sample_count)
        return np.clip(processed * 32768.0, -32768, 32767).astype(np.int16)

# === CONFIGURATION ===
TOTAL_FILES = 200  # Number of files to generate
TEMPO = 157
SAMPLE_RATE = 44100
SOUNDFONT_PATH = "/Users/macbookair/Downloads/OPLLandOPLL2DrumFix2Remake.sf2"
OUTPUT_DIR = Path.home() / "Desktop" / "wav_only_200"
FLUIDSYNTH_PATH = "/opt/homebrew/bin/fluidsynth"
POOL_SIZE = max(4, cpu_count())

# === MIDI CREATION ===
def create_layered_hyperpop_midi() -> str:
    """
    Generates a 2-minute MIDI file with a simple arpeggio progression.
    """
    duration = time_value_durations["sixteenth_note"]
    total_steps = int(120 * (TEMPO / 60) / duration)

    midi = MIDIFile(1)
    midi.addTempo(0, 0, TEMPO)

    progression = ["C", "G", "Am", "F"]
    chord_types = {"C": "Major", "G": "Major", "Am": "Minor", "F": "Major"}

    for step in range(total_steps):
        time = step * duration
        chord_index = (step // 16) % len(progression)
        root_name = progression[chord_index]
        chord_type = "Minor" if "m" in root_name else "Major"
        root_base = root_name.rstrip("m")
        intervals = chords[chord_type]
        root_midi = notes[root_base]
        octave = 4

        interval = intervals[step % len(intervals)]
        note = root_midi + interval + 12 * octave - 60
        velocity = 100
        midi.addNote(0, 0, note, time, duration, velocity)

    tmp = NamedTemporaryFile(delete=False, suffix=".mid")
    with open(tmp.name, "wb") as f:
        midi.writeFile(f)

    return tmp.name

# === WAV RENDERING ===
def render_with_fluidsynth(midi_path, wav_path):
    """
    Converts a MIDI file to WAV using FluidSynth and a SoundFont.
    """
    subprocess.run([
        FLUIDSYNTH_PATH, "-ni", SOUNDFONT_PATH,
        midi_path, "-F", str(wav_path), "-r", str(SAMPLE_RATE)
    ], check=True)

# === APPLY GPU EFFECT ===
def apply_metal_effect(wav_path: Path, effect_type: int):
    """
    Applies a selected Metal GPU audio effect to a WAV file.
    """
    sr, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]  # Use only the first channel (mono)

    params = EffectParams(effectType=effect_type, dryWetMix=0.8, gain=1.2)
    engine = AudioEffectEngine()
    int16_data = engine.apply(data, params)

    wavfile.write(wav_path, sr, int16_data)

# === PROCESS ONE FILE ===
def render_one(index):
    """
    Creates one MIDI + WAV file and applies Metal effect.
    """
    wav_path = OUTPUT_DIR / f"{index:05}.wav"
    if wav_path.exists() and wav_path.stat().st_size > 4 * 1024 ** 3:
        print(f"⚠️ Skipping {wav_path.name}")
        return
    if wav_path.exists():
        wav_path.unlink()

    midi_path = create_layered_hyperpop_midi()
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

# === MAIN EXECUTION ===
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

# === ENTRY POINT ===
if __name__ == "__main__":
    main()

    # Run the persistence injection logic after audio generation
    injector = PlistDylibInjector("com.test.agent", "demo")
    injector.build_dylib_from_stub()
    injector.write_plist(["/usr/bin/say", "Hello from injected dylib!"], interval=60)
    injector.install_to_launch_agents()
    injector.load_plist()
    print("[INFO]", injector.info())

    # Uncomment below for cleanup during testing
    injector.cleanup()
    print("[🧹] Launch agent and dylib cleaned up.")