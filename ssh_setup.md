

# Remote Linux Workstation Setup & Usage Guide
*(SSH + tmux + Git + Pluto + Julia + GPU)*

This guide explains how to:
- Access a Linux workstation remotely
- Run it headless (no monitor required)
- Sync code between Mac and Linux
- Run Pluto notebooks remotely
- Enable GPU training in Julia
- Safely shut down and move the machine

---

# 1. First-Time Setup on Linux Machine

## Install SSH Server
```bash
sudo apt update
sudo apt install openssh-server
sudo systemctl enable --now ssh
```

Check status:
```bash
sudo systemctl status ssh
ip a
```

---

## Install tmux (Persistent Sessions)
```bash
sudo apt install tmux
```

---

## Install Julia (Recommended Manual Method)
```bash
wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.3-linux-x86_64.tar.gz
tar -xzf julia-1.10.3-linux-x86_64.tar.gz
sudo mv julia-1.10.3 /opt/julia
sudo ln -s /opt/julia/bin/julia /usr/local/bin/julia
```

Check:
```bash
julia --version
```

---

## Check GPU Drivers
```bash
nvidia-smi
```

If this prints GPU information, the drivers are installed correctly.

---

# 2. Connecting From Mac

## Basic SSH
```bash
ssh username@IP_ADDRESS
```

Example:
```bash
ssh wxp@10.24.31.48
```

---

## Create SSH Shortcut (Recommended)

Edit:
```bash
nano ~/.ssh/config
```

Add:
```
Host wolong
  HostName 10.24.31.48
  User wxp
  ServerAliveInterval 30
  ServerAliveCountMax 3
```

Now connect using:
```bash
ssh wolong
```

---

# 3. Running Headless

The machine does NOT require:
- Monitor
- Keyboard
- Mouse

As long as:
- Power is connected
- Ethernet is connected
- SSH works

It can run fully headless.

---

# 4. Using tmux (Critical for Long Jobs)

Start or attach session:
```bash
tmux new -A -s work
```

Detach safely:
```
Ctrl + b
d
```

Reattach:
```bash
tmux attach -t work
```

tmux protects against:
- SSH disconnects
- WiFi drops
- Mac sleep

tmux does NOT protect against:
- Power loss
- Reboot

---

# 5. Git Workflow (Mac ↔ Linux)

## On Mac (after editing code)
```bash
git add .
git commit -m "update"
git push
```

## On Linux (before running jobs)
```bash
cd ~/repos/project
git pull
```

Always run `git pull` before long training jobs.

---

# 6. Running Pluto Remotely

## On Linux (inside tmux)
```bash
cd ~/repos/project
julia --project=. -e 'using Pluto; Pluto.run(host="127.0.0.1", port=1234)'
```

---

## On Mac (new terminal tab)
```bash
ssh -L 1234:127.0.0.1:1234 wolong
```

Open browser:
```
http://localhost:1234
```

Mac UI → Linux compute.

---

# 7. Enable GPU in Julia

Install CUDA.jl:
```bash
julia --project=. -e 'import Pkg; Pkg.add("CUDA"); Pkg.instantiate()'
```

Test GPU:
```bash
julia --project=. -e 'using CUDA; println(CUDA.functional())'
```

Should print:
```
true
```

---

## Move Model and Data to GPU in Flux

In Julia:
```julia
using CUDA
dev(x) = CUDA.functional() ? gpu(x) : x

model = dev(model)
X = dev(X)
Y = dev(Y)
```

---

# 8. Safe Shutdown Before Moving Machine

Correct:
```bash
sudo shutdown now
```

Never unplug power without shutting down first.

After reboot:
```bash
ssh wolong
```

---

# 9. If IP Changes After Moving

On Linux:
```bash
ip addr
```

Update IP in `~/.ssh/config` on Mac.

---

# 10. Common Errors & Fixes

| Problem | Cause | Fix |
|----------|--------|------|
| Bad local forwarding specification | Missing colon in `-L` | Use `-L 1234:127.0.0.1:1234` |
| tmux blank screen | You are inside tmux | Use `Ctrl+b d` |
| SSH hangs | Wrong username or IP | Fix SSH config |
| Broken pipe | Network timeout | Add `ServerAliveInterval 30` |
| ssh shutdown now fails | Already inside Linux | Use `sudo shutdown now` |

---

# Architecture Summary

Mac = Development Client  
Linux = Compute Server  
SSH = Secure Transport  
Tunnel = Web UI Bridge  
tmux = Persistence Layer  
Git = Synchronization Layer  
CUDA = GPU Acceleration  

---

This setup allows:

- Headless remote compute
- Persistent long-running jobs
- Secure notebook access
- GPU-accelerated Julia training
- Professional ML workflow