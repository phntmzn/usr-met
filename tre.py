def create_slow_tresillo_chords_midi() -> str:
    """
    Generates a 2-minute MIDI file with block chords in slow tresillo (triplet quarter) rhythm.
    """
    duration = time_value_durations["quarter_note"] * 2 / 3  # triplet quarter in beats
    beats_per_minute = TEMPO
    beats_per_second = beats_per_minute / 60
    total_beats = beats_per_minute * 2  # 2 minutes worth of beats
    total_steps = int(total_beats / duration)

    midi = MIDIFile(1)
    midi.addTempo(0, 0, TEMPO)

    progression = ["C", "G", "Am", "F"]
    chord_types = {"C": "Major", "G": "Major", "Am": "Minor", "F": "Major"}

    for step in range(total_steps):
        time = step * duration
        chord_index = (step // 3) % len(progression)
        root_name = progression[chord_index]
        chord_type = chord_types[root_name]
        
        if root_name.endswith("m"):
            root_base = root_name[:-1]
        else:
            root_base = root_name
        
        root_midi = notes[root_base]
        intervals = chords[chord_type]

        for interval in intervals:
            note = root_midi + interval + 12 * 4  # octave 4
            velocity = 100
            midi.addNote(0, 0, note, time, duration, velocity)

    tmp = NamedTemporaryFile(delete=False, suffix=".mid")
    with open(tmp.name, "wb") as f:
        midi.writeFile(f)

    return tmp.name
