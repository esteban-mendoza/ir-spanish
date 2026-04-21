
# External disk
cd /media/discoexterno/

lscpu # CPU info
free -h # Memory info
nvidia-smi # GPU info
df -h # Disk usage

# Check disk usage of the current directory
du -sh .[^.]* * 2>/dev/null | sort -hr

# Process management
ps aux | grep proyecto
ps -o user PID
kill <PIDs>
kill -9 <PIDs>

# Trigger processes in the background and redirect output to log files
cd /home/jmendoza/ir-spanish && nohup /home/jmendoza/miniconda3/envs/proyecto/bin/python -m baselines.bm25 > logs/bm25.log 2>&1 &

cd /home/jmendoza/ir-spanish && nohup /home/jmendoza/miniconda3/envs/proyecto/bin/python -m rerankers.fuse > logs/fuse-reranker-combnz-2-3.log 2>&1 &

# Check logs in real-time
tail -f /home/jmendoza/ir-spanish/logs/fuse-reranker-combnz-2-3.log

# TeX-related commands
brew install --cask basictex

sudo tlmgr update --self && sudo tlmgr update --all
tlmgr search --global --file mypackage.sty
sudo tlmgr install <package1> <package2>
tlmgr list --only-installed
