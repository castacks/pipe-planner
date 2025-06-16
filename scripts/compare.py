import numpy as np

def find_rectangles(diff_mask):
    visited = np.zeros_like(diff_mask, dtype=bool)
    rectangles = []
    H, W = diff_mask.shape

    for y in range(H):
        for x in range(W):
            if diff_mask[y, x] and not visited[y, x]:
                # flood‑fill this component
                stack, coords = [(y, x)], []
                visited[y, x] = True

                while stack:
                    cy, cx = stack.pop()
                    coords.append((cy, cx))

                    for ny, nx in ((cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)):
                        if (0 <= ny < H and 0 <= nx < W 
                                and diff_mask[ny, nx] 
                                and not visited[ny, nx]):
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                ys, xs = zip(*coords)
                rectangles.append((min(ys), max(ys)+1, min(xs), max(xs)+1))

    return rectangles

def main(old_path, new_path):
    old = np.load(old_path)
    new = np.load(new_path)
    assert old.shape == new.shape, "Arrays must be same shape"

    diff = old != new
    if not diff.any():
        print("No differences found.")
        return

    for y0, y1, x0, x1 in find_rectangles(diff):
        new_val = new[y0, x0]
        print(f"map[{y0}:{y1}, {x0}:{x1}] = {new_val}")
        print(f"valid[{y0}:{y1}, {x0}:{x1}] = {new_val}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python diff_npy_to_slices.py <old.npy> <new.npy>")
    else:
        main(sys.argv[1], sys.argv[2])
