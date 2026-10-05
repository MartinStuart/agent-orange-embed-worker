"""Crude 0.5-CPU emulator: runs a command and SIGSTOP/SIGCONTs it on a 50% duty
cycle (50 ms on / 50 ms off), approximating a CFS quota of 0.5 CPU. Pessimistic
for I/O (network waits are paused too). Local testing only."""
import os, signal, subprocess, sys, time
on, off = 0.05, 0.05
p = subprocess.Popen(sys.argv[1:])
try:
    while p.poll() is None:
        time.sleep(on)
        try: os.kill(p.pid, signal.SIGSTOP)
        except ProcessLookupError: break
        time.sleep(off)
        try: os.kill(p.pid, signal.SIGCONT)
        except ProcessLookupError: break
finally:
    try: os.kill(p.pid, signal.SIGCONT)
    except ProcessLookupError: pass
sys.exit(p.wait())
