# GaugeGuard Module Interface Contracts (DRAFT)

**Status:** Awaiting team review and confirmation  
**Author:** Prasoon  
**Date:** 2026-10-21  
**Purpose:** Define data types, function signatures, and conventions for interfaces between calibration.py, evaluation.py, vision.py, agent.py, and app.py.

---

## 1. Global Conventions

### 1.1 Image Representation
**EXISTING** (found in `vision.py:14`)
- **Type:** `numpy.ndarray` with `dtype=uint8`, shape `(H, W, 3)`, BGR channel order
- **Source:** `cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)`
- **Contract:** All vision functions accept/return BGR images as returned by `cv2.imread()` and `cv2.imdecode()`

### 1.2 Angle Convention
**PROPOSED** (needs team confirmation)
- **Type:** `float` degrees in range `[0, 360)`
- **Zero reference:** 0° = 12 o'clock position (straight up, negative y-axis in image coordinates)
- **Direction:** Clockwise positive (because image y-axis points down)
- **Formula:** `angle_deg = (math.degrees(math.atan2(needle_dx, -needle_dy)) % 360.0)`
  - Where `(needle_dx, needle_dy)` is the vector from dial center to needle tip
  - The negative sign on `dy` accounts for image coordinates (y increases downward)
- **Rationale:** Matches physical gauge dials where angles increase clockwise from 12 o'clock

**CONFLICT WITH EXISTING:** `calibration.py:22` documents angle as "degrees increase clockwise from the image's positive x axis (rightward)". This is inconsistent with typical gauge conventions. **NEEDS RESOLUTION.**

### 1.3 Units
**EXISTING** (found in `models.py:19`, `calibration.py:9`)
- Gauge values expressed in `profile.unit` (e.g., "PSI", "bar", "MPa", "°C")
- Unit is a string stored in `GaugeProfile.unit` and propagated to `ReadingEstimate.unit`

---

## 2. calibration.py Interface (Prasoon)

### 2.1 GaugeProfile
**EXISTING** (found in `calibration.py:7-13`)
```python
@dataclass(frozen=True)
class GaugeProfile:
    name: str
    unit: str
    minimum_value: float
    maximum_value: float
    minimum_angle_degrees: float
    maximum_angle_degrees: float
```

**PROPOSED EXTENSIONS** (needs confirmation):
```python
@dataclass(frozen=True)
class GaugeProfile:
    # EXISTING fields
    name: str
    unit: str
    minimum_value: float
    maximum_value: float
    minimum_angle_degrees: float
    maximum_angle_degrees: float
    
    # PROPOSED additions
    direction: str = "cw"  # "cw" or "ccw", future-proofing
    bounds_margin_deg: float = 5.0  # degrees from endpoints considered "at limit"
    operating_min: float | None = None  # safe operating range lower bound
    operating_max: float | None = None  # safe operating range upper bound
    agreement_tolerance_fraction: float = 0.02  # retry agreement: 2% of scale
    eval_tolerance_fraction: float = 0.05  # evaluation acceptance: 5% of scale
```

### 2.2 Functions

**PROPOSED:**

```python
def get_profile(name: str) -> GaugeProfile:
    """Retrieve a registered gauge profile by name.
    
    Raises:
        UnknownProfileError: Profile name not registered.
    """

def normalize_angle_deg(angle: float) -> float:
    """Normalize angle to [0, 360) range."""

def angle_to_value(
    angle_deg: float,
    profile: GaugeProfile,
    angle_uncertainty_deg: float | None = None
) -> CalibrationResult:
    """Convert needle angle to gauge reading with bounds checking.
    
    Args:
        angle_deg: Measured needle angle (0-360°, per convention).
        profile: Gauge calibration profile.
        angle_uncertainty_deg: Optional ±uncertainty in degrees.
    
    Returns:
        CalibrationResult with value, bounds flags, uncertainty range.
    
    Raises:
        CalibrationError: Invalid profile configuration (programmer error).
        
    Notes:
        - Bad measured data (None, NaN, out-of-range) returns flagged result
        - Never silently converts uncertain data to a reliable reading
    """

@dataclass(frozen=True)
class CalibrationResult:
    value: float | None  # None if angle invalid
    unit: str
    in_bounds: bool  # angle within profile's min/max sweep
    at_scale_limit: bool  # near endpoints (within bounds_margin_deg)
    within_operating_limits: bool  # in safe range if defined
    value_low: float | None  # lower bound if uncertainty provided
    value_high: float | None  # upper bound if uncertainty provided
    flags: tuple[str, ...]  # e.g., "out_of_bounds", "at_limit", "outside_operating"
    reason: str  # human-readable explanation
    
    def to_dict(self) -> dict: ...

def readings_agree(
    a: float | None,
    b: float | None,
    profile: GaugeProfile
) -> AgreementResult:
    """Check if two readings agree within profile tolerance.
    
    Args:
        a, b: Gauge readings in profile units. None => disagree.
        profile: Used for agreement_tolerance_fraction.
    
    Returns:
        AgreementResult with boolean, absolute difference, tolerance.
    """

@dataclass(frozen=True)
class AgreementResult:
    agree: bool
    difference: float | None  # absolute difference; None if either input None
    tolerance: float  # threshold used
    reason: str
    
    def to_dict(self) -> dict: ...
```

**Error Handling:**
- **EXISTING** (found in `calibration.py:23, 32, 38`): `ValueError` raised for invalid profile or out-of-range angles
- **PROPOSED:** `CalibrationError(ValueError)` for invalid profiles only; bad measured data returns `CalibrationResult(value=None, flags=[...], reason="...")`

**Dependencies:**
- **EXISTING** (confirmed in Prompt 1): No `cv2` import; stdlib + numpy only

---

## 3. vision.py Interface (Samiksha)

**EXISTING FUNCTIONS** (found in `vision.py`):

```python
def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an OpenCV BGR image."""

def inspect_image_quality(image_bytes: bytes) -> ImageQuality:
    """Return basic blur and contrast indicators."""

def preprocess_for_retry(image: np.ndarray, method: str) -> np.ndarray:
    """Apply documented preprocessing (clahe, adaptive_threshold)."""

def detect_dial_and_needle(image: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """TODO: implement and evaluate for the single supported gauge style."""
    # Currently raises NotImplementedError
```

### 3.1 Expected Extensions for Prasoon's Work

**PROPOSED** (needs Samiksha confirmation):

```python
@dataclass(frozen=True)
class NeedleDetectionResult:
    """Result from dial and needle detection."""
    success: bool
    angle_deg: float | None  # None if detection failed
    center: tuple[float, float] | None  # (x, y) dial center
    needle_tip: tuple[float, float] | None  # (x, y) needle endpoint
    confidence: float  # 0.0 to 1.0
    method: str  # e.g., "hough_lines", "contour_analysis"
    params: dict[str, Any]  # method parameters used
    quality_flags: tuple[str, ...]  # e.g., "low_confidence", "multiple_needles"
    annotated_image: np.ndarray | None  # BGR image with overlay
    reason: str
    
    def to_dict(self) -> dict: ...

def detect_dial_and_needle_full(
    image: np.ndarray,
    method: str = "default"
) -> NeedleDetectionResult:
    """Detect gauge dial and needle, return angle and metadata."""
```

**QUESTIONS FOR SAMIKSHA:**
1. Should `detect_dial_and_needle()` signature change from returning coordinate tuples to returning `NeedleDetectionResult`?
2. Or should there be a separate `detect_needle_angle()` function?
3. What method names will you use for retry attempts (e.g., "hough_lines_v1", "contour_edges", "template_match")?
4. Will you compute the angle internally, or should calibration.py provide a helper to compute angle from center + tip coordinates?
5. What quality flags will vision.py produce (beyond the existing "low_resolution", "possible_blur", "low_contrast")?

---

## 4. agent.py Interface (Kshitj)

**EXISTING FUNCTION** (found in `agent.py:7`):
```python
def inspect_upload(image_bytes: bytes) -> InspectionResult:
    """Initial orchestration scaffold; gauge detection and retries are TODOs."""
```

**EXISTING TYPES** (found in `models.py`):
```python
@dataclass(frozen=True)
class ReadingEstimate:
    value: float | None
    unit: str | None
    angle_degrees: float | None
    confidence: float
    status: str  # Currently: "needs_retake", "not_implemented"
    reason: str

@dataclass
class AgentStep:
    step_name: str
    method: str
    reason: str
    output_summary: dict[str, Any] = field(default_factory=dict)

@dataclass
class InspectionResult:
    reading: ReadingEstimate
    quality: ImageQuality | None
    action_log: list[AgentStep] = field(default_factory=list)
```

### 4.1 Proposed Status Values

**PROPOSED** (needs Kshitj confirmation):
- `"accepted"` — reading accepted with sufficient confidence
- `"needs_human"` — escalated for human review (uncertainty unresolved)
- `"rejected_input"` — image quality too poor to attempt reading
- `"error"` — internal processing failure

### 4.2 Expected Final Interface

**PROPOSED:**

```python
@dataclass
class InspectionResult:
    # EXISTING fields
    reading: ReadingEstimate
    quality: ImageQuality | None
    action_log: list[AgentStep]
    
    # PROPOSED additions
    needs_human: bool  # True whenever status != "accepted"
    attempts: list[dict]  # each attempt: {method, angle_deg, reading, flags, ...}
    uncertainty: dict | None  # {"level": "low|medium|high", "reasons": [...], "angle_disagreement_deg": float | None}
    annotated_image: np.ndarray | None  # final annotated result
    
    def to_dict(self) -> dict:
        """Return JSON-serializable dict (no numpy types)."""
```

### 4.3 Action Log Entry Structure

**PROPOSED** (JSON-serializable, per PDF requirements):

```python
{
    "step": int,  # 1-indexed step number
    "stage": str,  # "quality_check", "detection_attempt_1", "detection_attempt_2", "comparison", "decision"
    "tool": str,  # "opencv_laplacian", "hough_lines", "calibration", "agreement_check"
    "params": dict,  # method parameters used
    "input_summary": dict,  # e.g., {"image_size": [H, W], "preprocessed": true}
    "output_summary": dict,  # e.g., {"angle_deg": 127.3, "confidence": 0.82}
    "quality_flags": list[str],  # flags from this step
    "decision": str | None,  # "accept", "retry_different_method", "escalate_human", None if not terminal
    "reason": str,  # why this action was taken or why uncertain
    "duration_ms": float,
    "timestamp_utc": str  # ISO 8601 format
}
```

**QUESTIONS FOR KSHITJ:**
1. Should `InspectionResult` extend the existing structure or should there be a new `GaugeReadingResult` type?
2. How will you represent retry attempts — extend `action_log` or add a separate `attempts` field?
3. When two readings disagree, will you call `calibration.readings_agree()` or implement your own comparison?
4. What triggers retry: specific quality flags, low confidence threshold, or both?
5. How should calibration.py signal "uncertain but best-effort" vs "reject, do not use"?

---

## 5. Uncertainty Representation

### 5.1 Vision Uncertainty
**PROPOSED** (from vision.py):
- Confidence score: `float` in [0.0, 1.0]
- Quality flags: `list[str]` or `tuple[str, ...]`

### 5.2 Calibration Uncertainty
**PROPOSED** (from calibration.py):
- Provides `value_low` and `value_high` if `angle_uncertainty_deg` supplied
- Flags: `"out_of_bounds"`, `"at_limit"`, `"outside_operating"`

### 5.3 Agent Uncertainty
**PROPOSED** (from agent.py):
```python
{
    "level": "low" | "medium" | "high",
    "reasons": ["disagreement_exceeds_tolerance", "low_vision_confidence", "at_scale_limit"],
    "angle_disagreement_deg": float | None  # if retry performed
}
```

**Escalation Rule:** `needs_human = True` if uncertainty level is "high" or status != "accepted"

---

## 6. Action Log Requirements (from PDF)

**AGREED** (per competition PDF requirements):

Each action log entry must support reporting:
1. Method/tool name and parameters used
2. Output (angle, reading, confidence)
3. Quality flags raised
4. Why accepted or why uncertain
5. If retry: what method changed, what parameters changed
6. Comparison: difference between attempts
7. Decision and decision reason
8. Final status

The proposed action log structure in §4.3 covers all requirements.

---

## 7. Human Review Status

**PROPOSED:**
- `InspectionResult.needs_human: bool` is `True` whenever `ReadingEstimate.status != "accepted"`
- Status values requiring human review: `"needs_human"`, `"rejected_input"`, `"error"`

---

## 8. File Locations and Data Formats

### 8.1 Sample Images and Labels
**PROPOSED** (Prasoon's responsibility):
- Path: `sample_images/labels.csv`
- Format:
  ```csv
  filename,gauge_profile,ground_truth_value,unit,notes,consent_license
  gauge_01_clear.jpg,pressure_100psi,42.5,PSI,clear frontal view,CC0
  gauge_02_angled.jpg,pressure_100psi,67.8,PSI,30deg tilt,owner_consent
  ```
- All images stored in `sample_images/` (currently does not exist; will create)

### 8.2 Evaluation Results
**PROPOSED** (Prasoon generates):
- Path: `evaluation_results/` (gitignored, generated)
- Files: `metrics_summary.json`, `failure_cases.json`, `per_image_results.csv`

### 8.3 Gauge Profiles
**PROPOSED** (Prasoon maintains):
- Path: `gaugeguard/profiles.py` or `data/profiles.json`
- Contains profile definitions loaded by `get_profile()`

---

## 9. Error Handling Contracts

### 9.1 calibration.py
**EXISTING + PROPOSED:**
- **Invalid profile** (programmer error): raise `CalibrationError(ValueError)`
- **Bad measured data** (None, NaN, out-of-range angle): return `CalibrationResult(value=None, flags=[...], reason="...")`
- **Never:** silent conversion of uncertain data to reliable reading
- **Never:** raise exception for bad measured data from vision

### 9.2 vision.py
**PROPOSED** (needs Samiksha confirmation):
- **Invalid image bytes:** raise `ValueError` (EXISTING in `decode_image`)
- **Detection failure:** return `NeedleDetectionResult(success=False, angle_deg=None, ...)`
- **Never:** return fabricated angle when detection fails

### 9.3 agent.py
**PROPOSED** (needs Kshitj confirmation):
- **Processing error:** return `InspectionResult(..., status="error", reason="...", needs_human=True)`
- **Never:** hide failures or return status="accepted" with value=None

---

## 10. Open Questions by Teammate

### For Samiksha (vision.py):
1. Confirm angle convention: 0° at 12 o'clock or 3 o'clock (see §1.2 conflict)?
2. Should `detect_dial_and_needle()` change to return `NeedleDetectionResult`, or add a new function?
3. What method names will you use for retry attempts?
4. Will you compute angle from coordinates, or should calibration provide a helper?
5. What quality flags will vision produce beyond blur/contrast/resolution?
6. How should agent select which retry method to use (pass method name, or let vision decide)?

### For Kshitj (agent.py):
1. Should status values be `{"accepted", "needs_human", "rejected_input", "error"}` or different?
2. Extend `InspectionResult` or create new `GaugeReadingResult` type?
3. How to represent retry attempts in the result structure?
4. Will you use `calibration.readings_agree()` or implement comparison logic?
5. What triggers retry: specific flags, confidence threshold, or both?
6. How should calibration signal "uncertain but best-effort" vs "reject entirely"?
7. When should agent call `angle_to_value()` with `angle_uncertainty_deg` vs without?

### For Bhumika (app.py):
1. Should `app.py` call `agent.inspect_upload()` or a different entry point?
2. What UI elements needed for displaying uncertainty, action log, and human-review status?
3. Should annotated images show angle overlay, or just quality indicators?
4. How to display action log: expandable JSON, timeline, or summary?

---

## 11. Next Steps

1. **Angle convention resolution:** Team decides 0° = 12 o'clock vs 3 o'clock (URGENT)
2. **Samiksha + Prasoon:** Agree on `detect_dial_and_needle()` return type and angle computation responsibility
3. **Kshitj + Prasoon:** Agree on retry comparison (use `readings_agree()` or separate logic)
4. **Kshitj:** Finalize `InspectionResult` structure and status values
5. **Prasoon:** Implement extended `GaugeProfile`, `CalibrationResult`, `AgreementResult` after confirmation
6. **All:** Review and sign off on this draft, update to CONTRACTS.md

---

**Review Status:** DRAFT — awaiting teammate confirmation on PROPOSED items.
