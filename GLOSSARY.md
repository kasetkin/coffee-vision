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
A photo in the classifier's training pool (`data/segcropped/pool`, one since ticket ML-8; one per camera before), labelled with its country.
_Avoid_: Main dataset

**Segmenter positive**:
A positive the owner chose for the segmenter because its setup differs from the pools (framing, container, distance).
_Avoid_: OOD positive

**Example photo**:
A photo the page offers a visitor to try the service with, in place of their own. It may be a positive or a negative. It is never in any training or evaluation set, so no model the visitor's answer comes from has seen it.
_Avoid_: Sample, demo photo, fixture (a fixture is a photo the deploy checks a release with)

**Canonicalization**:
Turning a photo into its **canonical photo**: the one form both models read, in training and in serving alike, whatever format, colour space or orientation the photo arrived in. Not the scaling of model inputs by a mean and standard deviation, which keeps the name "normalization".
_Avoid_: Normalization (of a photo), preprocessing, re-encoding (for the whole step)

**Accepted format**:
A photo format the service promises to read, named on the page and to agents. A photo in any other format is refused unread, even when a decoder for it exists.
_Avoid_: Supported format, decodable format

**Fixture**:
A photo the deploy runs through every release to check it, fixed until the deploy's own checks change. An example photo is never a fixture, even though the deploy checks its answer too: the examples are the page's, and they may change on their own.
_Avoid_: Test photo, example photo

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

## Publishing

**Leak**:
Personal data in a file the repo publishes: a public IP address, an email address, a credential, identifying photo metadata, or a match of one of the owner's private patterns. Infrastructure names (SSH aliases, the DVC remote URL, the VM's user and hostname, the live site's domain) are not leaks.
_Avoid_: Secret (that means a credential only), private data (for a published leak)
