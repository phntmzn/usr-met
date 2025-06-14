def render_one(index):
    """
    Creates one MIDI + WAV file and applies Metal effect with musically valid key and progression.
    """
    wav_path = OUTPUT_DIR / f"{index:05}.wav"
    if wav_path.exists() and wav_path.stat().st_size > 4 * 1024 ** 3:
        print(f"⚠️ Skipping {wav_path.name}")
        return
    if wav_path.exists():
        wav_path.unlink()

    # Randomly select a key (major or minor)
    keys = list(notes.keys())
    # Only use natural notes for root to avoid double sharps/flats in progression
    root_key = random.choice(keys)
    mode = random.choice(["Major", "Minor"])

    # Generate I - V - vi - IV progression (or its minor equivalent)
    # keys list is assumed to be chromatic (C, C#, D, D#, ... B)
    root_index = keys.index(root_key)
    if mode == "Major":
        # I, V, vi, IV in the selected key
        progression = [
            root_key,  # I
            keys[(root_index + 7) % len(keys)],  # V
            keys[(root_index + 9) % len(keys)],  # vi
            keys[(root_index + 5) % len(keys)]   # IV
        ]
        chord_types = ["Major", "Major", "Minor", "Major"]
    else:
        # i, v, III, VI in the selected minor key
        progression = [
            root_key,  # i
            keys[(root_index + 7) % len(keys)],  # v
            keys[(root_index + 3) % len(keys)],  # III
            keys[(root_index + 5) % len(keys)]   # VI
        ]
        chord_types = ["Minor", "Minor", "Major", "Major"]

    def create_musical_midi(progression, chord_types, mode):
        duration = time_value_durations["quarter_note"] * 2 / 3
        total_steps = int(120 * (TEMPO / 60) / duration)
        midi = MIDIFile(1)
        midi.addTempo(0, 0, TEMPO)
        for step in range(total_steps):
            time = step * duration
            chord_index = (step // 3) % len(progression)
            root_name = progression[chord_index]
            chord_type = chord_types[chord_index]
            root_midi = notes[root_name.rstrip("m")]
            intervals = chords[chord_type]
            for interval in intervals:
                note = root_midi + interval + 12 * (3 - 4)
                midi.addNote(0, 0, note, time, duration, 100)
        tmp = NamedTemporaryFile(delete=False, suffix=".mid")
        with open(tmp.name, "wb") as f:
            midi.writeFile(f)
        return tmp.name
