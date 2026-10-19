# Evaluation plan

## Label set

Use one supported gauge design. Record the verified reading, unit, image conditions, and whether a human would accept the image. Keep calibration/tuning images separate from final evaluation images.

Include clear, rotated, blurry, glare, and partially obstructed examples. Record the source and permission/license for every image.

## Metrics

- Mean absolute error for accepted readings.
- Share of accepted readings within a predefined tolerance.
- Incorrect-acceptance rate, with the denominator explicitly stated.
- Retry rate and human-escalation rate.
- Local and AWS latency under documented conditions.

## Agentic evidence

For each retry case, preserve the first result, the quality signal, the selected alternate OpenCV operation, the second result, and the final accept/escalate decision. Include at least one unresolved failure that correctly ends in human review.
