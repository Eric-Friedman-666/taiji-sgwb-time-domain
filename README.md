# Paper data v1.0.0

This branch stores the compact numerical archive for the Taiji SGWB paper. The 14 binary parts concatenate to `lisa_td_paper_replay_data_20261017.tar.gz` (56,274,616 bytes; SHA256 `858b8b70c7ea53cc1340cbf55c24fbfeb4dc8594475772bee7312a3b1b736d9f`).

For automatic download, assembly, checksum verification and extraction, use `python get_data.py` from [the main code branch](https://github.com/Eric-Friedman-666/taiji-sgwb-time-domain/tree/main). Then run `python reproduce.py all --output outputs` to regenerate all 15 figures and five tables.

The archive contains numerical data and archived posterior/evidence outputs, with no rendered figures. The optional large inference-input archive is not published here.

For manual assembly from a checkout of this branch:

```bash
cat lisa_td_paper_replay_data_20261017.tar.gz.part-* > lisa_td_paper_replay_data_20261017.tar.gz
shasum -a 256 -c SHA256SUMS.txt
```
