# Report: A4 Paper Detection as a Search Problem

Section B of the COMP1032 Fundamentals of AI coursework. I designed and implemented this section myself. The photo dataset was collected by my coursework group (Group 23).

*Adapted from my coursework report for GitHub. Student IDs are removed and the images are relabelled. The results sections (7.2 to 11) have been updated to match the measured output of the notebook and include a visual check of the detected outlines, and the h2 analysis in section 5.2 is corrected.*

## 1. Problem formulation

The goal is to find an A4 sheet of paper in a real photo where the sheet is partly covered by a hand or an arm. Instead of using the nested loops in the original A4 detection script, I restate detection as an explicit search problem and use A* to find the 4 corners of the sheet in a search space of candidate intersection points.

The system runs as a 3-stage pipeline:

- **Image processing.** A Gaussian blur (7×7 kernel) removes noise, then Canny edge detection and dilation strengthen the edges.
- **Hough line transform.** The scikit-image Hough transform finds up to 25 dominant lines in the edge image. The intersections of roughly perpendicular line pairs (angle difference of at least 50°) become candidate corner points, and points that are close together are merged with a clustering function.
- **Search.** The candidate corners form the search space. The search uses geometric constraints and heuristics to find the set of four points that best approximates the boundary of the sheet. The provided `likelihoodA4` function scores the final rectangle by combining edge alignment with a prior on the A4 aspect ratio.

## 2. State definition

### 2.1 State representation

A state is an ordered sequence of k corner points, s = [p0, …, pk−1], where k is 0 to 4. Each point pi = (x, y) is a candidate intersection point from the Hough step.

### 2.2 Initial and goal states

The initial state is the empty list (k = 0): no corners chosen. The goal test is `is_goal()`, which returns `k == 4`. A goal state is only accepted if:

- `corners_span_image()` holds: the bounding box covers at least 25% of the image width and height, and
- its `likelihoodA4` score is above the minimum of 0.20.

### 2.3 Node structure

Each state is stored in a `SearchNode` with:

- `k`: the number of corners chosen (the depth)
- `corners`: the ordered list of chosen candidate points
- `g_cost`: the accumulated path cost
- `parent`: the parent node, for tracing the path

The visited set uses `frozenset(corners)` as the state key, so the same set of corners is recognised as a duplicate whatever order it was found in.

## 3. Successor function

`get_successors` expands a state by adding one new candidate corner, with constraints that depend on the depth.

### 3.1 First corner (k = 0)

From the empty state, any candidate point can be the first corner, so n successors are generated.

### 3.2 Later corners (k = 1, 2, 3)

These constraints apply cumulatively:

- **Line connection.** The new point must lie on a Hough line that passes through the current last corner, i.e. |x cos θ + y sin θ − ρ| < 88 pixels. This makes consecutive corners joined by detected edges.
- **Geometric consistency.** Once three or more corners are chosen, the interior angle at the middle of the last three points must be between 75° and 105°. This enforces roughly right-angled corners, as on a rectangular sheet.
- **Ordering.** Later corners must have (x + y) greater than or equal to that of the first corner. This avoids counting the same quadrilateral several times from different starting corners and roughly halves the search space.

### 3.3 Fourth corner

Adding the fourth corner applies further checks:

- **A4 ratio bounds.** Every side must be at least 175 pixels, opposite sides may differ by at most 50%, and the aspect ratio must be between 0.8 and 2.2 to allow for perspective distortion.
- **Closing check.** The fourth corner must lie on a Hough line that passes through the first corner, so the quadrilateral is closed.
- **Fourth angle.** The angle at the fourth corner (between corners 3, 4 and 1) must also be between 75° and 105°.

## 4. Cost function

### 4.1 Choice: penalty-based step cost

I use a uniform step cost plus an angle penalty: g(n) = depth + accumulated angle penalty. Each added corner whose interior angle is not 90° adds a penalty of |angle − 90| / 15.

### 4.2 Justification

- **It separates goal states.** All goal states have the same depth, so with a purely uniform cost they would tie. With the penalty, a near-perfect rectangle collects a lower g-cost than a skewed quadrilateral, so the search prefers geometrically better solutions.
- **It works with the heuristics.** The penalty sits in g(n), the cost so far, while h(n) estimates the cost still to come. Because the smallest possible penalty per step is 0, h1(n) = 4 − k never exceeds the remaining cost, so h1 stays admissible.
- **It is robust.** Thanks to the filtering, most chosen corners are close to 90°, so the penalty is usually under 0.1 per corner. This keeps g(n) close to the depth while still ordering the priorities.

## 5. Heuristic design

### 5.1 h1(n): remaining corners (admissible)

h1(n) = 4 − k, where k is the number of corners chosen. It estimates the remaining cost as the number of steps still needed to reach a goal state.

**Admissibility.** A heuristic is admissible if h(n) ≤ h\*(n) for every n. Each action adds exactly one corner at a cost of at least 1 (the minimum penalty is 0), so from a state with k corners the true remaining cost is at least 4 − k. Since h1(n) = 4 − k, it never overestimates.

**Consistency.** A heuristic is consistent if h(n) ≤ c(n, a, n′) + h(n′) for every successor n′. Adding a corner increases k by 1, so h1(n′) = h1(n) − 1, and the minimum step cost is 1. Therefore h1(n) = 1 + h1(n′) ≤ c(n, a, n′) + h1(n′). The triangle inequality holds, so h1 is consistent.

### 5.2 h2(n): promise score (admissible but not consistent)

h2(n) = (4 − k) × (1 − confidence), where confidence = 0.8 × angle confidence + 0.2 × edge confidence.

- **Angle confidence (weight 0.8).** For k ≥ 3, this measures how close the interior angle at the last three corners is to 90°. A perfect 90° gives 1.0, falling linearly to 0 at 15° away. Below k = 3 it is a neutral 0.5.
- **Edge confidence (weight 0.2).** The share of edge pixels in a 70×70 patch around the last corner added. A corner on a strong edge gives a higher density and therefore more confidence.

**Admissibility.** Both confidence terms are between 0 and 1, so the confidence is too, and h2(n) = (4 − k) × (1 − confidence) ≤ 4 − k = h1(n) ≤ h\*(n). So h2 never overestimates: it is admissible. (My original report described h2 as inadmissible; that was wrong.)

**Why h2 is not consistent.** Consistency needs h2(n) ≤ c(n, a, n′) + h2(n′). Take a state with k = 2 and confidence 0, so h2 = 2, and a successor with k = 3 and confidence 1, so h2 = 0. The step cost is about 1, and 2 > 1 + 0, so the inequality fails. Because the search keeps a visited set and never reopens a state, an inconsistent heuristic means A\* with h2 loses its guarantee of returning the lowest-cost path first.

The idea behind h2 was to push the search toward geometrically promising branches. In practice, because h2 is often much smaller than h1, it gives the search less guidance (see section 10).

## 6. Completeness and optimality

Each image gives a finite set of candidate points. All three methods (BFS, A\* with h1, A\* with h2) search this finite space up to a node limit (50,000 for BFS, 20,000 for each A\* run). There are at most N⁴ states, where N is the number of candidate points (0 to 49 per image in this dataset; the Hough step found no candidates at all in two images).

Completeness therefore depends on the preprocessing. If the Hough transform cannot see the paper's edges (heavy occlusion, poor lighting, or low contrast between paper and background), no valid set of four corners exists in the search space and the search fails.

On optimality: with h1 (admissible and consistent), A\* expands the lowest-cost path first. The search also stops early at the first goal scoring at least 0.40, trading strict optimality for speed. A\* with h2 is admissible but not consistent, so with a visited set it has no optimality guarantee. In my code, "BFS" orders the queue by path cost g(n) with h = 0, so it behaves as uniform-cost search: it finds a solution if one exists within the node budget, but not necessarily the highest-scoring rectangle.

## 7. Complexity analysis

### 7.1 Theoretical complexity

Let N be the number of clustered candidate points and b the branching factor after pruning.

**Without pruning (worst case).** The search tree has 4 levels, one per corner, with N, N − 1, N − 2 and N − 3 choices. That gives N × (N − 1) × (N − 2) × (N − 3) = O(N⁴) ordered 4-tuples.

**With pruning.** The successor constraints cut the branching factor sharply. Line connectivity limits each expansion to points on the Hough lines through the current corner, and the 75° to 105° angle filter removes geometrically incompatible candidates. In practice b is much smaller than N, so the effective search space is O(b⁴).

**Time and space.** BFS and A\* with h1 have the same worst case, O(b⁴), but A\* expands fewer nodes in practice thanks to its priority ordering. The priority queue and visited set also take O(b⁴) space.

### 7.2 Empirical comparison

Measured over the 25 images where a rectangle was returned (from the notebook's per-image output):

| Metric | BFS | A\* h1 | A\* h2 |
| --- | --- | --- | --- |
| Node limit | 50,000 | 20,000 | 20,000 |
| Mean nodes expanded | 678 | 225 | 568 |
| Median nodes expanded | 448 | 187 | 401 |
| Max nodes expanded | 3,715 | 712 | 3,144 |
| Mean time per image | 11.9 s | 3.2 s | 10.7 s |
| Mean score | 0.474 | 0.474 | 0.474 |

No run came close to its node limit, and all three methods reached the same score on every one of these images. (The notebook prints scores, not the corner coordinates of each method's result, so I have not checked that the corners themselves are identical.)

## 8. Experimental results

### 8.1 Detection summary

The dataset has 30 photos, 5 each from six people (sets A to F). Set A is my own and is the only set included in this repository.

A rectangle was returned for 25 of the 30 images (83%). However, a visual check of the drawn outlines shows that about 10 of those 25 closely match the whole sheet. The other 15 cover only part of the sheet (often one half, cut off where an arm crosses it), are shifted, or are skewed. Whole-sheet detection is therefore closer to a third of the images.

| Image | Rectangle returned | Score | Px/cm | Outline (visual check) |
| --- | --- | --- | --- | --- |
| A1 | Yes | 0.412 | 46.673 | Close to whole sheet |
| A2 | Yes | 0.434 | 38.605 | Left half only |
| A3 | Yes | 0.455 | 57.410 | Skewed |
| A4 | Yes | 0.476 | 54.640 | Shifted |
| A5 | Yes | 0.613 | 40.696 | Narrow strip on the right |
| B1 | Yes | 0.530 | 31.427 | Partial |
| B2 | Yes | 0.567 | 50.242 | Partial (top) |
| B3 | Yes | 0.518 | 33.641 | Partial (bottom) |
| B4 | Yes | 0.418 | 36.122 | Partial (top) |
| B5 | No | – | – | – |
| C1 | No | – | – | – |
| C2 | Yes | 0.566 | 54.157 | Close to whole sheet |
| C3 | Yes | 0.498 | 60.306 | Shifted |
| C4 | No | – | – | – |
| C5 | No | – | – | – |
| D1 | Yes | 0.456 | 65.392 | Close to whole sheet |
| D2 | Yes | 0.415 | 68.574 | Close to whole sheet |
| D3 | Yes | 0.412 | 43.211 | Partial |
| D4 | Yes | 0.418 | 39.083 | Partial (left) |
| D5 | Yes | 0.431 | 44.436 | Partial (top) |
| E1 | Yes | 0.425 | 65.737 | Close to whole sheet |
| E2 | Yes | 0.560 | 66.773 | Close to whole sheet |
| E3 | Yes | 0.452 | 45.346 | Partial (top) |
| E4 | Yes | 0.425 | 65.737 | Close to whole sheet |
| E5 | Yes | 0.495 | 46.275 | Partial |
| F1 | No | – | – | – |
| F2 | Yes | 0.504 | 44.587 | Partial (top) |
| F3 | Yes | 0.437 | 69.877 | Close to whole sheet |
| F4 | Yes | 0.426 | 69.531 | Close to whole sheet |
| F5 | Yes | 0.509 | 68.571 | Close to whole sheet |

The visual check is my own judgement from the overlay images, so treat it as approximate.

## 9. Best and worst cases

### 9.1 The score does not track correctness

The five highest scores were A5 (0.613), B2 (0.567), C2 (0.566), E2 (0.560) and B1 (0.530). Only C2 and E2 of these are close to the whole sheet; A5, the top score, outlines only a narrow strip. `likelihoodA4` rewards rectangles whose sides lie on strong edges with an A4-like shape, and a smaller rectangle that follows the paper's edge on one side and the edge of an arm or a printed label on the other can score as well as, or better than, the true outline.

The images that worked best had the sheet held flat and fairly straight, with good contrast against the background and little of the sheet's boundary hidden.

### 9.2 Failures

No rectangle was returned for B5, C1, C4, C5 and F1. In C1 and C5 the Hough step produced no candidate corners at all. In the other three it produced 7 to 23 candidates, but no set of four passed all the geometric constraints: all three methods ran out of states to expand (29 to 610 nodes) well before reaching any node limit, so the failures are not caused by the node budget. Common causes were a sheet that was bent or curved, a hand covering most of the sheet, and backgrounds with strong straight lines that the Hough transform picked up instead of the paper's edges.

### 9.3 Failure patterns seen in the overlays

- **An arm across the middle of the sheet (set E, images 3 and 5).** The arm lies horizontally across the paper and splits it into an upper and lower half. The Hough transform correctly finds the strong horizontal edge of the arm's shadow, and the search snaps to only one half of the sheet. The same pattern explains many of the partial outlines in section 8.1.
- **Creased paper (set F, image 2).** Scratches and fold lines on the paper create many short spurious edges inside the sheet in the Canny output.
- **A steep camera angle with an arm across the sheet (set D, images 2 to 4).** Perspective makes the corners' angles in the image very different from 90°, which pushes many true corner candidates outside the 75° to 105° filter.

## 10. Heuristic comparison

- **A\* with h1 was the clear winner on efficiency.** It expanded about a third as many nodes as BFS on average (225 vs 678) and was about 3.7 times faster (3.2 s vs 11.9 s per image), while reaching the same score. It is also admissible and consistent.
- **A\* with h2 gave little gain over BFS** (568 nodes and 10.7 s on average). Its confidence term shrinks the heuristic on promising branches, which makes A\* rely more on g(n) and behave more like uniform-cost search.
- **BFS** (h = 0) expanded the most nodes but, within its budget, reached the same scores.

Because the search stops at the first goal scoring at least 0.40, and the geometric constraints leave few valid rectangles, all three methods reached the same score. The differences were in how much work it took to get there.

| Property | BFS | A\* h1 | A\* h2 |
| --- | --- | --- | --- |
| Admissible? | n/a (h = 0) | Yes | Yes |
| Consistent? | n/a (h = 0) | Yes | No |
| Complete? | Within node budget | Within node budget | Within node budget |
| Optimal? | Lowest-cost path within budget | Yes | No guarantee |
| Nodes expanded (mean) | 678 | 225 | 568 |
| Score (mean) | 0.474 | 0.474 | 0.474 |

## 11. Conclusion

I modelled A4 detection as an explicit state-space search and ran BFS, A\* with h1 and A\* with h2 in a common pipeline. The geometric constraints in the successor function cut the effective branching factor from N to a much smaller b, and A\* with the admissible h1 heuristic reached the same scores as the other methods with about a third of the node expansions.

The main weakness is in detection quality, not in the search. A rectangle was returned for 25 of 30 images, but only about 10 closely match the whole sheet, and the score function does not reliably separate correct outlines from partial ones. Possible improvements:

- Reward rectangles that cover more of the bright paper region, or penalise outlines that cross strong interior edges such as an arm
- Adapt the preprocessing parameters to each image, or use a multi-scale approach
- Use a learned edge or segmentation model to handle bent paper and cluttered backgrounds
- Evaluate against hand-labelled corner positions instead of the score alone
