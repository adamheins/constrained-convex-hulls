# Constrained Convex Hulls

## Usage
To generate an SVG showing a constrained convex hull, clone this repo and run:
```
# generates cch.svg
uv run scripts/cch.py

# generate a different shape by using a different random seed
uv run scripts/cch.py --seed 314159

# change lower and/or upper bounds on the convex hull weights
uv run scripts/cch.py --lower-bound 0.03 --upper-bound 0.7
```
