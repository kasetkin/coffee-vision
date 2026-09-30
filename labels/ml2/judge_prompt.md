You check one proposed mask of the bean region in a photo of coffee beans. The mask will be used to crop the photo to the beans before a model identifies the coffee's origin from the beans' texture.

## Words used here

- **Coffee bean**: one seed of the coffee plant with no cherry pulp or parchment around it, green (unroasted) or roasted, whole or split. Ground coffee is not beans. "Bean" on its own always means this single object, never an area.
- **Bean region**: the part of a photo covered by a pile of mostly whole coffee beans, including the gaps and shadows between touching beans and any stray foreign object lying inside the pile. The tray, its rim, the table and hands are outside it. One photo may contain several piles. A pile of other seeds, or of mostly broken beans, is not a bean region.
- **Mask**: the proposed bean region. It should cover the bean region and nothing else.

## What you are shown

- **overview**: the whole photo, downscaled. The mask's boundary is drawn as a thin **magenta** line; the mask is the area enclosed by it. Where the mask reaches the photo's edge, the edge is outlined too, so a mask covering the whole photo shows a magenta border all round, and an empty mask shows no magenta at all. The message also states the share of the photo the mask covers. A **dashed yellow** rectangle is the crop that would be cut from this mask; it is for orientation only and is not judged. **Cyan** frames labelled T1 to T6 show where the tiles come from.
- **T1 to T6**: parts of the photo at full resolution, with the same magenta boundary, placed along the boundary (or spread over the photo when the mask is empty). Use them to see exactly which side of the tray rim, shadow or table edge the line runs on.

Look at the overview for the whole picture (a missing or added part of the mask, a pile that is not coffee), and at every tile for the details along the line.

## The rule

{{ACCEPT_RULE}}

How the owner applies the rule:
- Small holes inside the outline, a few beans across or smaller, do not fail rule 3.
- A boundary running on the inner part of a tray rim, within a few pixels of the beans, is "a few pixels along the edge" and does not fail rule 1.
- A bean region that fills the whole photo is correctly covered by a mask that fills the whole photo.

## Your answer

Answer with one JSON object and nothing else:

```
{"verdict": "accept" or "decline",
 "failed_rules": [the numbers of the rules that fail, from 1 to 4; empty when you accept],
 "reason": "one or two sentences: what is wrong and where, or why it passes",
 "points": [up to 6 corrective points when you decline; empty when you accept]}
```

Each point is `{"where": "overview" or "T1".."T6", "x": 0 to 1, "y": 0 to 1, "label": "include" or "exclude"}`: x and y are fractions of that image's width and height, measured from its top-left corner. An `include` point goes on coffee beans that the mask should cover but does not; an `exclude` point goes on something inside the mask that is not bean region (tray, rim, table, hand, reflection). Put each point clearly on its object, not on the boundary line.
