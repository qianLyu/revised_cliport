"""Ravens tasks."""

from cliport.tasks.align_box_corner import AlignBoxCorner
from cliport.tasks.assembling_kits import AssemblingKits
from cliport.tasks.assembling_kits import AssemblingKitsEasy
from cliport.tasks.assembling_kits_seq import AssemblingKitsSeqSeenColors
from cliport.tasks.assembling_kits_seq import AssemblingKitsSeqUnseenColors
from cliport.tasks.assembling_kits_seq import AssemblingKitsSeqFull
from cliport.tasks.block_insertion import BlockInsertion
from cliport.tasks.block_insertion import BlockInsertionEasy
from cliport.tasks.block_insertion import BlockInsertionNoFixture
from cliport.tasks.block_insertion import BlockInsertionSixDof
from cliport.tasks.block_insertion import BlockInsertionTranslation
from cliport.tasks.manipulating_rope import ManipulatingRope
from cliport.tasks.align_rope import AlignRope
from cliport.tasks.packing_boxes import PackingBoxes
from cliport.tasks.packing_shapes import PackingShapes
from cliport.tasks.packing_boxes_pairs import PackingBoxesPairsSeenColors
from cliport.tasks.packing_boxes_pairs import PackingBoxesPairsUnseenColors
from cliport.tasks.packing_boxes_pairs import PackingBoxesPairsFull
from cliport.tasks.packing_google_objects import PackingSeenGoogleObjectsSeq
from cliport.tasks.packing_google_objects import PackingUnseenGoogleObjectsSeq
from cliport.tasks.packing_google_objects import PackingSeenGoogleObjectsGroup
from cliport.tasks.packing_google_objects import PackingUnseenGoogleObjectsGroup
from cliport.tasks.palletizing_boxes import PalletizingBoxes
from cliport.tasks.place_red_in_green import PlaceRedInGreen
from cliport.tasks.put_block_in_bowl import PutBlockInBowlSeenColors
from cliport.tasks.put_block_in_bowl import PutBlockInBowlUnseenColors
from cliport.tasks.put_block_in_bowl import PutBlockInBowlFull
from cliport.tasks.stack_block_pyramid import StackBlockPyramid
from cliport.tasks.stack_block_pyramid_seq import StackBlockPyramidSeqSeenColors
from cliport.tasks.stack_block_pyramid_seq import StackBlockPyramidSeqUnseenColors
from cliport.tasks.stack_block_pyramid_seq import StackBlockPyramidSeqFull
from cliport.tasks.sweeping_piles import SweepingPiles
from cliport.tasks.separating_piles import SeparatingPilesSeenColors
from cliport.tasks.separating_piles import SeparatingPilesUnseenColors
from cliport.tasks.separating_piles import SeparatingPilesFull
from cliport.tasks.task import Task
from cliport.tasks.towers_of_hanoi import TowersOfHanoi
from cliport.tasks.towers_of_hanoi_seq import TowersOfHanoiSeqSeenColors
from cliport.tasks.towers_of_hanoi_seq import TowersOfHanoiSeqUnseenColors
from cliport.tasks.towers_of_hanoi_seq import TowersOfHanoiSeqFull
from cliport.tasks.pick_place_blocks_middle import PickPlaceMiddleUnseenColors
from cliport.tasks.pick_place_blocks_middle import PickPlaceMiddleSeenColors
from cliport.tasks.pick_place_blocks_middle import PickPlaceMiddleFull
from cliport.tasks.put_block_in_bowl_spatial import PutBlockInBowlSpatialUnseenColors
from cliport.tasks.put_block_in_bowl_spatial import PutBlockInBowlSpatialSeenColors
from cliport.tasks.put_block_in_bowl_spatial import PutBlockInBowlSpatialFull
from cliport.tasks.pretrain_semantic_spatial import PretrainSemanticSpatialUnseenColors
from cliport.tasks.pretrain_semantic_spatial import PretrainSemanticSpatialSeenColors
from cliport.tasks.pretrain_semantic_spatial import PretrainSemanticSpatialFull
from cliport.tasks.pretrain_semantic_spatial_left import PretrainSemanticSpatialLeftUnseenColors
from cliport.tasks.pretrain_semantic_spatial_left import PretrainSemanticSpatialLeftSeenColors
from cliport.tasks.pretrain_semantic_spatial_left import PretrainSemanticSpatialLeftFull
from cliport.tasks.ssp_hard import SSPHardUnseenColors
from cliport.tasks.ssp_hard import SSPHardSeenColors
from cliport.tasks.ssp_hard import SSPHardFull

names = {
    # demo conditioned
    'align-box-corner': AlignBoxCorner,
    'assembling-kits': AssemblingKits,
    'assembling-kits-easy': AssemblingKitsEasy,
    'block-insertion': BlockInsertion,
    'block-insertion-easy': BlockInsertionEasy,
    'block-insertion-nofixture': BlockInsertionNoFixture,
    'block-insertion-sixdof': BlockInsertionSixDof,
    'block-insertion-translation': BlockInsertionTranslation,
    'manipulating-rope': ManipulatingRope,
    'packing-boxes': PackingBoxes,
    'palletizing-boxes': PalletizingBoxes,
    'place-red-in-green': PlaceRedInGreen,
    'stack-block-pyramid': StackBlockPyramid,
    'sweeping-piles': SweepingPiles,
    'towers-of-hanoi': TowersOfHanoi,

    # goal conditioned
    'align-rope': AlignRope,
    'assembling-kits-seq-seen-colors': AssemblingKitsSeqSeenColors,
    'assembling-kits-seq-unseen-colors': AssemblingKitsSeqUnseenColors,
    'assembling-kits-seq-full': AssemblingKitsSeqFull,
    'packing-shapes': PackingShapes,
    'packing-boxes-pairs-seen-colors': PackingBoxesPairsSeenColors,
    'packing-boxes-pairs-unseen-colors': PackingBoxesPairsUnseenColors,
    'packing-boxes-pairs-full': PackingBoxesPairsFull,
    'packing-seen-google-objects-seq': PackingSeenGoogleObjectsSeq,
    'packing-unseen-google-objects-seq': PackingUnseenGoogleObjectsSeq,
    'packing-seen-google-objects-group': PackingSeenGoogleObjectsGroup,
    'packing-unseen-google-objects-group': PackingUnseenGoogleObjectsGroup,
    'put-block-in-bowl-seen-colors': PutBlockInBowlSeenColors,
    'put-block-in-bowl-unseen-colors': PutBlockInBowlUnseenColors,
    'put-block-in-bowl-full': PutBlockInBowlFull,
    'stack-block-pyramid-seq-seen-colors': StackBlockPyramidSeqSeenColors,
    'stack-block-pyramid-seq-unseen-colors': StackBlockPyramidSeqUnseenColors,
    'stack-block-pyramid-seq-full': StackBlockPyramidSeqFull,
    'separating-piles-seen-colors': SeparatingPilesSeenColors,
    'separating-piles-unseen-colors': SeparatingPilesUnseenColors,
    'separating-piles-full': SeparatingPilesFull,
    'towers-of-hanoi-seq-seen-colors': TowersOfHanoiSeqSeenColors,
    'towers-of-hanoi-seq-unseen-colors': TowersOfHanoiSeqUnseenColors,
    'towers-of-hanoi-seq-full': TowersOfHanoiSeqFull,
    'pick-place-middle-unseen-colors': PickPlaceMiddleUnseenColors,
    'pick-place-middle-seen-colors': PickPlaceMiddleSeenColors,
    'pick-place-middle-full': PickPlaceMiddleFull,
    'put-block-in-bowl-spatial-seen-colors': PutBlockInBowlSpatialSeenColors,
    'put-block-in-bowl-spatial-unseen-colors': PutBlockInBowlSpatialUnseenColors,
    'put-block-in-bowl-spatial-full': PutBlockInBowlSpatialFull,
    'pretrain-semantic-spatial-seen-colors': PretrainSemanticSpatialSeenColors,
    'pretrain-semantic-spatial-unseen-colors': PretrainSemanticSpatialUnseenColors,
    'pretrain-semantic-spatial-full': PretrainSemanticSpatialFull,
    'pretrain-semantic-spatial-left-seen-colors': PretrainSemanticSpatialLeftSeenColors,
    'pretrain-semantic-spatial-left-unseen-colors': PretrainSemanticSpatialLeftUnseenColors,
    'pretrain-semantic-spatial-left-full': PretrainSemanticSpatialLeftFull,
    'ssp-hard-seen-colors': SSPHardSeenColors,
    'ssp-hard-unseen-colors': SSPHardUnseenColors,
    'ssp-hard-full': SSPHardFull,
}
