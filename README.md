# Constrained Convex Hulls

## Install

Install system dependencies:
```
sudo apt install libcdd-dev libgmp-dev
```

Clone this repo and build using [uv]():
```
git clone https://github.com/adamheins/constrained-convex-hulls
cd constrained-convex-hulls
uv sync
```

## Usage

Generate constrained convex hull examples as SVGs:
```
# generates cch.svg
uv run scripts/cch.py

# generate a different shape by using a different random seed
uv run scripts/cch.py --seed 314159

# change lower and/or upper bounds on the convex hull weights
uv run scripts/cch.py --lower-bound 0.03 --upper-bound 0.7
```

Generate examples of polygons:
```
uv run scripts/poly.py
```

Generate an example of applying uniform padding to a polygon:
```
uv run scripts/padding.py
```
