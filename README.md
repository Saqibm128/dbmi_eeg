# EEG for dbmi

## Setup
### Sacred
Project uses Sacred to keep track of experiments, results, and stores in local MongoDB or hosted MongoDB. Just comment out to just use the script without anything.
```
sacred.observers.append(obs)
```
### Config.json
Use a config.json to choose file paths.
``` config.json
{
  "preprocessed_1": "/mnt/c/Users/sawer/src/dbmi/data_pp1/data/pp_1",
  "preprocessed_2": "/mnt/c/Users/sawer/src/dbmi/data_pp1/data/pp_1"
}

```
