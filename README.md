# airway_seg

Generally important information:
- 0 -> patient
- 1 -> row
- 2 -> column
- 3 -> slice
- 4 -> channel

Hounsfield unit are -2000-2000 so it's a nice number
- regarding standardizing image...
- ONLY divide IMAGE (not mask)

# 22 March 2022

Ran into a bit of trouble regarding the data generator... how do we decide what passes as a patch?

Does Kyle's approach not work?
- Makes sense to not use `argmax`, but at the other end of things if there's only one voxel in the whole patch does that suffice?
- What should the threshold be if this is the case?

Validated the Data Generator with Kyle.
- Now need to make it a function.
- ideally next deliverable is a full single run.

