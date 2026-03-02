# Running Jobs on Sherlock

## Every time: sync code and submit

**1. Establish SSH connection (activates ControlMaster for rsync):**
```bash
ssh sherlock
```

**2. In a new local terminal, push code changes:**
```bash
rsync -av --delete --exclude='.venv' --exclude='data/' /Users/miagarvey/disease-drug-networks/ sherlock:~/disease-drug-networks/
```

**3. Back on Sherlock, create a job script:**
```bash
cat > run_myjob.sh
```
Paste content, then `Ctrl+D` to save:
```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=results/myjob_%j.out

module load python/3.12.1
cd $HOME/disease-drug-networks
source .venv/bin/activate
python scripts/my_script.py
```

**4. Submit and monitor:**
```bash
sbatch run_myjob.sh
squeue -u $USER              # PD=pending, R=running
tail -f results/myjob_*.out  # watch output live
```

**5. When done, pull results locally:**
```bash
rsync -av sherlock:~/disease-drug-networks/results/ /Users/miagarvey/disease-drug-networks/results/
```

---

## One-time setup (already done, only redo if venv is lost)

```bash
cd ~/disease-drug-networks
module load python/3.12.1
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2.0" --only-binary=numpy
pip install torch-geometric
pip install pandas tqdm scikit-learn --only-binary=:all:
```

Data symlinks (already done):
```bash
ln -s /scratch/users/mgarvey1/kg.csv ~/disease-drug-networks/data/kg.csv
ln -s /scratch/users/mgarvey1/primekg.pkl ~/disease-drug-networks/data/primekg.pkl
```
