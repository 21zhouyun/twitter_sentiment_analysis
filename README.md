# Prereq
- Go to https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/access to learn how to access NUS computing cluster
- remove the '?' character in train test data's header otherwise it will cause pandas to throw decoding error
- create python venv and install requirements.txt
- sbatch train.sh to train the model on slurm machines. See train_template.sh for an example
- you might want to checkout sftp to sync code between your local machine and slurm

# Research Plan
-~~ basic implementation on slurm~~
- try twitter specific bert model
- aux sentiment analysis data?