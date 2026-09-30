# Coffee Vision

Identifying a coffee's origin from a phone photo of its beans.

## Language

**Coffee bean**:
One seed of the coffee plant with no cherry pulp or parchment around it, green (unroasted) or roasted, whole or split. Ground coffee is not beans. "Bean" on its own always means this single object, never an area.
_Avoid_: Coffee cherry (the whole fruit, pulp included), parchment coffee

**Bean region**:
The part of a photo covered by a pile of mostly whole coffee beans, including the gaps and shadows between touching beans and any stray foreign object lying inside the pile. The tray, its rim, the table and hands are outside it. One photo may contain several. A pile of other seeds, or of mostly broken beans, is not a bean region.
_Avoid_: Bean pixels, beans (for an area), tray region

**Negative**:
A photo with no bean region in it. A **pile-like negative** shows a pile of something bean-like that is not coffee beans (other seeds, grains, legumes, ground coffee, coffee cherries).
_Avoid_: Background photo, junk photo

## Mask quality

**Judge**:
The Claude model that looks at a proposed bean-region mask and gives a verdict, pass or fail, plus corrective points on a fail.
_Avoid_: Reviewer, grader

**Base mask**:
A mask the owner looked at at full resolution and accepted as correct. Planted defects are made from it.
_Avoid_: Ground truth, gold mask

**Planted defect**:
A deliberate, known error introduced into a base mask to measure how often the judge catches it.
_Avoid_: Synthetic error, corruption

**Catch rate**:
The share of planted defects the judge fails.

**False-fail rate**:
The share of unmodified base masks the judge fails.
