"""Process resource counters that work on Linux (/proc) and macOS (/dev/fd, ps): file descriptors, OS threads, resident memory."""
import os
import subprocess


def fd_count():
    return len(os.listdir("/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"))  # both listings include the descriptor used to list them


def os_threads():
    if os.path.isdir("/proc/self/task"):
        return len(os.listdir("/proc/self/task"))
    return len(subprocess.check_output(["ps", "-M", "-p", str(os.getpid())], text=True).splitlines()) - 1  # one row per thread after the header


def rss_mb():
    if os.path.exists("/proc/self/statm"):
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) / 1024
