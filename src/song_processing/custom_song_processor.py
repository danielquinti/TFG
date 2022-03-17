class CustomSongProcessor(SongProcessor):
    def read_song(self, filepath):
        try:
            return libGPFile.GPFile.read(filepath)
        except EOFError:
            raise FileParsingException

    def get_duration(self, beat):
        return int.from_bytes(beat.duration, byteorder='big', signed=True) + 2

    def is_chord(self, beat):
        return sum(x is not None for x in beat.strings) > 1

    def is_rest(self, beat):
        g_string = __find_match_idx__(beat.strings, lambda x: x is not None)
        return g_string is None or beat.strings[g_string].noteType is None

    def get_valid_track(self, song):
        idx = __find_match_idx__(song.tracks, lambda x: re.search(self.track_name, x.name, flags=re.IGNORECASE))
        track = song.tracks[idx]
        if idx is not None and track.numStrings == 6:
            song.dropAllTracksBut(idx)
            return song.beatLists
        return None

    def compute_note(self, beat):
        string_to_base_note = {6: 4, 5: 9, 4: 2, 3: 7, 2: 11, 1: 4}
        note_mod = 12
        g_string = __find_match_idx__(beat.strings, lambda x: x is not None)
        base_note = string_to_base_note[g_string]
        offset = beat.strings[g_string].noteType[1]
        return (base_note + offset) % note_mod

    def get_measure_list(self, track):
        return track

    def get_beat_list(self, measure):
        return measure[0]
