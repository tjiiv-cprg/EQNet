---
machines:
    3090-C:
        argoverse_v2_dataroot: /mnt/data/argoverse2
---

## Notice
- Read Extra-Experiment-Conditions (EEC) file, it defines everything the program and the developer need to reproduce an experiment. It is a markdown file with a metadata region. The meta data is for the machine, and the markdown is for the developer.
- Working directory is fixed in the root directory here, every path in EEC files should be a full path w.r.t. the root directory.