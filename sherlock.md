ssh [sunet]@login.sherlock.stanford.edu
srun -p gpu --gres=gpu:1 --time=02:00:00 --mem=32G --cpus-per-task=4 --pty bash