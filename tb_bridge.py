"""Parse scorer-train stdout lines into TensorBoard events (no touch to the live run)."""
import re, sys
from torch.utils.tensorboard import SummaryWriter
w = SummaryWriter("/work/out/tb/scorer-27b")
seen, last_step = set(), 0
for raw in sys.stdin.buffer:
    line = raw.decode("utf-8", "ignore")
    m = re.search(r"ep(\d+) step (\d+)/(\d+) loss ([\d.]+) ([\d.]+)s/step", line)
    if m:
        g = int(m.group(2)); last_step = max(last_step, g)
        if ("loss", g) not in seen:
            seen.add(("loss", g))
            w.add_scalar("train/loss", float(m.group(4)), g)
            w.add_scalar("train/sec_per_step", float(m.group(5)), g); w.flush()
    m = re.search(r"== eval (step(\d+)|epoch\d+-end): val Spearman ([+-][\d.]+)", line)
    if m:
        g = int(m.group(2)) if m.group(2) else last_step
        if ("eval", g) not in seen:
            seen.add(("eval", g)); w.add_scalar("val/spearman", float(m.group(3)), g); w.flush()
