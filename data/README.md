# DementiaBank Data Notes

## Source dataset

- https://media.talkbank.org/dementia/English/0extra/ADReSS-2020/ADReSS-IS2020-train.zip?f=save
- https://media.talkbank.org/dementia/English/0extra/ADReSS-2020/ADReSS-IS2020-test.zip?f=save

## Metadata files

- `cc_meta_data.txt`: train control metadata (`label` forced to `0` by script)
- `cd_meta_data.txt`: train dementia metadata (`label` forced to `1` by script)
- `2020Labels.txt`: test metadata (`label` read directly from file)

## Data folders

- `data/audio`: source subject audio (`Sxxx.wav`)
- `data/transcript`: source CHAT transcripts (`Sxxx.cha`)

## How IDs are used

`extract_par_data.py` joins transcript/audio/metadata by subject ID (e.g., `S001`, `S160`) and writes:

- `outputs/par_only/train.csv` (IDs from `cc_meta_data.txt` + `cd_meta_data.txt`)
- `outputs/par_only/test.csv` (IDs from `2020Labels.txt`)