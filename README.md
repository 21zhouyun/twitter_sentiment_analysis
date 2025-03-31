# Prereq
- Go to https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/access to learn how to access NUS computing cluster
- remove the '?' character in train test data's header otherwise it will cause pandas to throw decoding error
- create python venv and install requirements.txt
- you might want to checkout sftp to sync code between your local machine and slurm
- download train test file into ./archive

Project structure:
- archive
    -  train.csv
    -  test.csv
- train.py
- train.sh

# trianing
Modify train.sh as needed. Then:

```bash
sbatch train.sh
tail -f slurm-XXX.out
```


# Research Plan
-~~ basic implementation on slurm~~
- try twitter specific bert model
- aux sentiment analysis data?