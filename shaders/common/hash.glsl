// Spatial hashing utilities

// Hash a 2D cell coordinate to a table index
uint hash_cell(ivec2 cell, uint table_size) {
    // Cantor-style hash with bit mixing
    uint h = uint(cell.x) * 73856093u ^ uint(cell.y) * 19349663u;
    return h & (table_size - 1u); // table_size must be power of 2
}

// Convert world position to cell coordinate
ivec2 pos_to_cell(vec2 pos, float cell_size, vec2 world_min) {
    vec2 local = pos - world_min;
    return ivec2(floor(local / cell_size));
}

// Hash a world position directly
uint hash_position(vec2 pos, float cell_size, vec2 world_min, uint table_size) {
    ivec2 cell = pos_to_cell(pos, cell_size, world_min);
    return hash_cell(cell, table_size);
}
