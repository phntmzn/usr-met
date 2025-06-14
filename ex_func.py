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
