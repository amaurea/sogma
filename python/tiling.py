# Tiled map support. With this all the maps will be tiled, even
# when not using distributed maps, since that's what the GPU wants.
# We need to distinguish between 3-4 different layouts:
# 1. The target map geometry. Some subset of the full sky
# 2. The equivalent full sky geometry. Determines wrapping
# 3. The tiling. Tiles the full sky geometry, but can
#    overshoot to allow for deferred wrapping. Given by a tile
#    size and the number of global y and x tiles. Only
#    a subset of these tiles are ever actually used, so having
#    a big overshoot isn't expensive. 2x overshoot in x
#    should be a very conservative choice.
# 4. The tile ownership. A two-way list of tiles "owned" by
#    this mpi task. Used to avoid double-counting during
#    CG and reductions, since the same tile will be in
#    several mpi tasks' workspaces.
# 5. The local wrapped tiles. A two-way list of tiles
#    workspace tiles after wrapping
# 6. The lowal workspace tiles. A two-way list of the tiles
#    we will actually work on.
# We also want to precompute transfer information, including
# 1. global buffer size, offsets and indices into owned tile list.
#    a) buffer size and offsets in byte units. Requires us to know
#       the number of pre-dims and the bit depth.
#    b) agnostic. size and offsets multiplied by npre*nbyte_per_num
#       on-the-fly
#    Let's go with b), since the multiplication will take microseconds
#    for these short arrays.
# 2. local buffer size and indices into wrapped tiles
# 3. information needed for translating between work tiles and
#    wrapped tiles, if necessary.
# We will store this in the class Tiling, which won't concern
# itself with the actual data, just how it's laid out.

# Building this structure takes some prerequisites.
# 1. Given the map geometry, derive the fullsky geometry
# 2. Read in a list of tods. For each tod we should have at least a
#    ra,dec bounding box as part of the tod database.
# 3. Translate the bounding boxes to pixels. Add multiples of wrap
#    until min(x) >= 1
# 4. Define the global tiling
# 5. Distribute TOD ownership to mpi tasks using the bounding boxes and
#    per-tod weights like nsamp. This could be done by calculating the
#    mid-point of each bounding box, and then splitting by [ra], [ra,dec]
#    or [rise/set,ra,dec]. This step is hard to do optimally. We want
#    each to be as local as possible
# 6. Distribute tile ownership. This could be done blindly, but that
#    will require more communication than necessary. A Better version
#    would be to base it on the tod ownership and bounding boxes.
#    For very wide scans, it may be worth it to separate rising and
#    setting, since the tiles would be in a narrow curved parallelogram,
#    but this requires better bounding boxes than just a rectangle for the
#    local workspaces. We can add this later. The function that does this
#    would need the tiling, tod list and tod metdata, and tod ownerships.
#    It may be best to have the same function do both #5 and #6
# 7. Determine the local work tiles. This can be as simple as getting
#    the bounding box of the bounding boxes of our tods, and turning that
#    into tile lists.
# 8. Determine the wrapped local tiles from the local work tiles and
#    tiling and wrap. The number of wrapped tiles may be different from
#    the work tiles.
# 9. Build the communication buffer info
