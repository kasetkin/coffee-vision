# Coffee Vision

Identifying a coffee's origin from a phone photo of its beans.

## Language

**Coffee bean**:
One seed of the coffee plant with no cherry pulp or parchment around it, green (unroasted) or roasted, whole or split. Ground coffee is not beans. "Bean" on its own always means this single object, never an area.
_Avoid_: Coffee cherry (the whole fruit, pulp included), parchment coffee

**Bean region**:
The part of a photo covered by a pile of mostly whole coffee beans, including the gaps and shadows between touching beans and any stray foreign object lying inside the pile. The tray, its rim, the table and hands are outside it. One photo may contain several. A pile of other seeds, or of mostly broken beans, is not a bean region. Always written "the photo's bean region", never "region" on its own.
_Avoid_: Bean pixels, beans (for an area), tray region, region

**Country**:
The country a coffee was grown in, by name only (e.g. Ethiopia). Since ticket ML-3 it is the class: what the model predicts and is scored on.

**Farm region**:
The growing area inside a country, written Country, Region[, Subregion] (e.g. Ethiopia, Yirgacheffe, Kochere). It means the same as "origin". It may be unknown, leaving the country only. A fact about a photo, never a label.
_Avoid_: Region (on its own), lot

**Misc**:
Anything else known about a coffee, kept apart from its country and farm region in `classes.txt`: grade (AA, Excelso), variety (PinkBourbon), species (Robusta), or a farm, brand or washing-station name (LaPastora, TataNahual, MonteCristo, Minca, Gisuma). Never a label.
_Avoid_: Lot

**Negative**:
A photo with no bean region in it. A **pile-like negative** shows a pile of something bean-like that is not coffee beans (other seeds, grains, legumes, ground coffee, coffee cherries).
_Avoid_: Background photo, junk photo

**Positive**:
A photo with at least one bean region in it; the counterpart of a negative.

**Pool photo**:
A photo in one of the classifier's training pools, labelled with its country.
_Avoid_: Main dataset

**Segmenter positive**:
A positive the owner chose for the segmenter because its setup differs from the pools (framing, container, distance).
_Avoid_: OOD positive

## Mask quality

**Judge**:
Whoever gives a verdict, pass or fail, on a proposed bean-region mask: the owner, or a Claude model.
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
