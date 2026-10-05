# Formula Student Driverless: Path Planning Solution Document

This document explains the design, algorithm, and testing methodology implemented for the Path Planning assignment (Parts 1 & 2).

---

## 1. Problem Overview

In Formula Student Driverless (FSAI), the racecar must navigate through an unknown track marked by colored cones:
- **Yellow Cones (`color == 0`)**: Define the **RIGHT** boundary of the track.
- **Blue Cones (`color == 1`)**: Define the **LEFT** boundary of the track.
- **Car Pose `(x, y, yaw)`**: The car starts at world coordinate $(x, y)$ pointing along heading angle `yaw` (in radians).

**Mission**: Generate a collision-free, drivable path between 5 and 10 meters long with step size $\le 0.5\text{m}$ that keeps the car safely centered between the left (blue) and right (yellow) track boundaries.

---

## 2. Algorithm Architecture (Part 1)

The pipeline is split into three simple stages:

```
[Detected Cones + Car Pose]
            │
            ▼
    1. Centerline Waypoint Extraction
       ├── Both sides visible?  ──> Compute gate midpoints
       ├── One side visible?    ──> Offset boundary perpendicularly by track half-width (1.5m)
       └── No cones visible?    ──> Extrapolate straight along car heading
            │
            ▼
    2. Dynamic Turning Apex (Scenario 14 & Sharp Angles)
       └── If |car heading - target heading| > 65°:
           Inject sweeping turning apex to ensure vehicle turning radius is respected
            │
            ▼
    3. B-Spline Curve Smoothing (scipy.interpolate.splprep)
       └── Fit order-3 B-spline with weighted anchors and roughness penalty
            │
            ▼
    4. Uniform Resampling (step = 0.1m)
       └── Yields silky smooth 8.5m path without polygonal edges
```

### 2.1 Centerline Waypoint Extraction
- **Both sides visible (`n_blue >= 1` and `n_yellow >= 1`)**:
  Opposite cones form gates across the track. The midpoint $M = \frac{\text{Blue} + \text{Yellow}}{2}$ gives the exact geometric center. The forward track tangent $\vec{T}_{\text{track}}$ is perpendicular to the vector from Yellow to Blue (rotated $90^\circ$ clockwise).
- **Single side visible (Yellow only or Blue only)**:
  When one boundary is occluded, the track direction is estimated from the visible boundary (or car heading), and the centerline is estimated by offsetting perpendicularly by the assumed track half-width ($1.5\text{m}$):
  - Yellow boundary $\implies$ offset to the **left** by $1.5\text{m}$.
  - Blue boundary $\implies$ offset to the **right** by $1.5\text{m}$.
- **Zero cones visible**:
  The planner seeds a straight path along the car's current heading.

### 2.2 Vehicle Kinematics & Sharp Turns (Scenario 14)
In Scenario 14, the car starts facing down-right at $\text{yaw} = -62^\circ$ while the gate is at $(3.0, 3.5)$ heading $+X$. Connecting these directly would require an impossible $16\text{ cm}$ turning radius that exceeds physical vehicle steering limits.
- **Solution**: When $|\Delta \theta| > 65^\circ$, the algorithm automatically computes an expanded turning apex along the initial heading. The car executes a wide, gentle left-hand arc before leveling into the gate, keeping curvature well within physical vehicle capabilities.

### 2.3 B-Spline Curve Smoothing (`splprep`)
Rather than relying on piecewise linear segments, the sequence of centerline waypoints is smoothed using **Parametric B-Spline Smoothing** (`scipy.interpolate.splprep`):
- Clamped weights anchor the path to the car's initial position and heading.
- A curvature/roughness penalty ($s = 0.04$) minimizes steering jerk.
- Dense sampling ($\text{step} = 0.1\text{m}$) produces 86 smoothly spaced points, ensuring no polygonal kinks appear in visualization or vehicle control.

---

## 3. Part 2: Handling Three or More Cones on One Side

### 3.1 Approach
When given 3 cones on one side (e.g. 3 yellow cones or 3 blue cones), the track boundary is no longer just a straight line—it may curve into a bend or chicane.

To leverage all 3 cones:
1. **Multi-Point Tangent Estimation**:
   For any sequence of cones $C_0, C_1, \dots, C_{n-1}$ sorted along the track, the boundary tangent at each cone is computed using centered differences:
   $$\vec{u}_i = \frac{C_{i+1} - C_{i-1}}{\|C_{i+1} - C_{i-1}\|}$$
2. **Perpendicular Normal Offsetting**:
   - For Yellow cones: rotate $90^\circ$ CCW $\implies \vec{n}_i = (-u_{i,y}, u_{i,x})$
   - For Blue cones: rotate $90^\circ$ CW $\implies \vec{n}_i = (u_{i,y}, -u_{i,x})$
   - Waypoint: $M_i = C_i + \text{half\_width} \cdot \vec{n}_i$
3. **Adaptive Width Measurement**:
   If a cone is also detected on the opposite side, the actual track width is measured at the nearest gate and used instead of the default $1.5\text{m}$.
4. **B-Spline Smoothing**:
   All 3 waypoints are passed to the B-spline smoother, capturing the true curve of the boundary.

### 3.2 Why This Solution Was Chosen
- **Generalization**: Works seamlessly for 2, 3, 4, or $N$ cones without code branching.
- **Curvature Awareness**: Captures progressive bends (e.g. constant radius curves and chicanes) that a 2-point line approximation misses.
- **Physical Feasibility**: Spline smoothing filters out noise while honoring boundary constraints.

### 3.3 Limitations & Future Work
- **Assumed Track Width**: When only one side has cones, the planner assumes a standard $3.0\text{m}$ track width ($1.5\text{m}$ half-width). If track width narrows or widens in that section, the estimated centerline may shift towards one edge.
- **Cone Association / Sorting**: Cones are sorted by Euclidean distance from the car. In complex chicanes or overlapping switchbacks, Delaunay triangulation or Voronoi diagram pairing is recommended to prevent misordering.
- **Field-of-View Range**: The planner operates on local detections up to 5–6 meters ahead. For high-speed racing, integration with SLAM / global mapping is necessary for lookahead horizons $>15\text{m}$.

---

## 4. Test Scenarios (1 to 25)

The test suite in `src/scenarios.py` contains 25 comprehensive scenarios:
- **Scenarios 1–20**: Baseline single/dual gate scenarios, varying car poses and heading angles.
- **Scenario 21**: 3 Yellow cones on a straight track.
- **Scenario 22**: 3 Blue cones on a straight track.
- **Scenario 23**: 3 Blue cones defining a curved left turn.
- **Scenario 24**: 3 Yellow cones defining a curved right turn / diagonal bend.
- **Scenario 25**: 3 Yellow cones on right + 1 Blue cone on left (testing measured width + 3-cone curvature).

### Test Results
```text
All 25 scenarios: PASSED
- Max step size: 0.10m (requirement: <= 0.5m) [PASS]
- Path length: 8.5m - 8.8m (requirement: 5m - 10m) [PASS]
- Collision avoidance: Safely centered within all gates [PASS]
```

---

## 5. How to Run

1. Activate virtual environment:
   ```powershell
   .\.venv\Scripts\activate
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run any scenario (e.g., Scenario 2, 14, 23, 25):
   ```powershell
   python -m src.run --scenario 2
   python -m src.run --scenario 14
   python -m src.run --scenario 23
   python -m src.run --scenario 25
   ```
