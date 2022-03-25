# airway_seg

# 22 March 2022

Ran into a bit of trouble regarding the data generator... how do we decide what passes as a patch?

Does Kyle's approach not work?
- Makes sense to not use `argmax`, but at the other end of things if there's only one voxel in the whole patch does that suffice?
- What should the threshold be if this is the case?

Validated the Data Generator with Kyle.
- Now need to make it a function.
- ideally next deliverable is a full single run.
