import argparse
import numpy as np
from os.path import join
from moisesdb.dataset import MoisesDB
from moisesdb.defaults import mix_4_stems
from moisesdb.utils import save_audio

MOISESDB_SONGS_MISSINGSTEM = [
    "46bc5393-7753-44ae-913b-bd5fa8f33e98",
    "b92cb1ca-baa9-4c74-b6dc-36389671ed76",
    "ee082817-dbda-4fbf-b5aa-8dce2320ae35",
    "0358fd1e-244a-4422-9a42-29b5d68f6e4b",
    "7dd515b0-e218-425d-b8bf-a75056237d6a",
]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_base", type=str, default="data/moisesdb")
    parser.add_argument("--dir_4stems", type=str, default="data/moisesdb_4stems")
    parser.add_argument("--sample_rate", type=int, default=44100)
    args = parser.parse_args()

    dir_base = args.dir_base
    dir_4stems = args.dir_4stems
    sample_rate = sample_rate

    # Load the database
    db = MoisesDB(data_path=dir_base, sample_rate=sample_rate)

    # Loop over tracks
    for i in range(len(db)):

        # Load track
        track = db[i]
        track_id = track.id

        # Ignore the tracks for which some stem is missing
        if track_id in MOISESDB_SONGS_MISSINGSTEM:
            continue

        # Define out dir
        out_path = join(dir_4stems, track_id)

        # Load stems and assemble into 4-stems format
        stems = track.stems
        stems = track.mix_stems(mix_4_stems)

        # Get min length to have all stems with the same length
        min_len = min(stems[k].shape[-1] for k in stems.keys())

        # Record stems and build mixture
        mixtures = np.zeros((2, min_len))
        for stem_name, samples in stems.items():
            samples = samples[..., :min_len]  # Trim to min_len
            save_audio(
                join(out_path, f"{stem_name}.wav"),
                samples,
                sr=sample_rate,
            )
            mixtures += samples

        # Record mixture
        save_audio(
            join(out_path, f"mixture.wav"),
            mixtures,
            sr=sample_rate,
        )

# EOF
