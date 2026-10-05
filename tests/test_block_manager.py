import pytest

from pagedbatch.block_manager import BlockAllocator, BlockTable, NoFreeBlocks


def test_allocate_and_free_round_trip():
    alloc = BlockAllocator(num_blocks=4, block_size=8)
    assert alloc.num_free == 4
    blocks = alloc.allocate(3)
    assert len(set(blocks)) == 3
    assert alloc.num_free == 1 and alloc.num_used == 3
    alloc.free(blocks[:2])
    assert alloc.num_free == 3
    with pytest.raises(ValueError):
        alloc.free(blocks[:1])  # double free
    alloc.free(blocks[2:])
    assert alloc.num_free == 4


def test_exhaustion_raises():
    alloc = BlockAllocator(num_blocks=2, block_size=4)
    alloc.allocate(2)
    assert not alloc.can_allocate(1)
    with pytest.raises(NoFreeBlocks):
        alloc.allocate(1)


def test_block_table_slots_follow_block_layout():
    alloc = BlockAllocator(num_blocks=8, block_size=4)
    table = BlockTable(alloc)
    assert table.blocks_needed(1) == 1
    assert table.blocks_needed(4) == 1
    assert table.blocks_needed(5) == 2
    slots = table.append(6)  # two blocks
    assert len(table.blocks) == 2 and table.num_tokens == 6
    b0, b1 = table.blocks
    assert slots == [b0 * 4 + i for i in range(4)] + [b1 * 4 + i for i in range(2)]
    assert table.blocks_needed(2) == 0  # fits in the partial block
    assert table.blocks_needed(3) == 1
    more = table.append(3)
    assert len(table.blocks) == 3
    assert more[:2] == [b1 * 4 + 2, b1 * 4 + 3]
    assert table.internal_fragmentation == 3 * 4 - 9
    assert table.slots(0, 9) == slots + more
    with pytest.raises(IndexError):
        table.slot(9)
    table.free()
    assert alloc.num_free == 8 and table.num_tokens == 0 and table.blocks == []
