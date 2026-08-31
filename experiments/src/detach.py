"""Run a command in a new session so it survives the launching shell.

Long fetches and sweeps outlive the shell that starts them. `nohup` only ignores
SIGHUP, which is not enough: when the session is torn down the whole process
group gets signalled and the job dies with it. Leaving the process group via
setsid(2) is the actual fix. macOS exposes it through os.setsid but ships no
/usr/bin/setsid binary, hence this helper.

Standard double fork: the first child calls setsid() to become session and group
leader, the second fork guarantees the surviving process is not a session leader
and so can never reacquire a controlling terminal.

Usage:
    python detach.py --log FILE -- COMMAND [ARGS...]

Prints the detached pid to stdout, then returns immediately.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="file to receive stdout and stderr")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="command to run, preceded by --")
    args = ap.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("no command given", file=sys.stderr)
        return 2

    log_path = os.path.abspath(args.log)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Hand the pid back through a pipe: after the double fork the caller is no
    # longer an ancestor of the surviving process, so waitpid cannot report it.
    read_fd, write_fd = os.pipe()

    if os.fork() > 0:
        os.close(write_fd)
        with os.fdopen(read_fd) as r:
            print(r.read().strip())
        return 0

    os.close(read_fd)
    os.setsid()

    if os.fork() > 0:
        os._exit(0)

    with os.fdopen(write_fd, "w") as w:
        w.write(str(os.getpid()))

    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    os.dup2(fd, 1)
    os.dup2(fd, 2)

    os.execvp(cmd[0], cmd)
    os._exit(127)  # only reached if exec fails


if __name__ == "__main__":
    sys.exit(main())
