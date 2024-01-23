# nightshade

nightshade for linux

special thanks to the folk at Sand Lab, University of Chicago for the original nightshade
code which they so graciously neglected to actually PyArmor.

## Installation

```bash
git clone <this repo>
cd nightshade
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel setuptools-scm
python -m pip install -e '.[all]'
```

I make no promises as to whether this actually works, or whether the dependencies are even correct.

CLI isn't implemented yet either but if I ever get around to it you should be able to just run `nightshade --help`

GLHF!

### License

MIT because I can't find any license info in the original code.
