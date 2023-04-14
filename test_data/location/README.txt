data files adapted from http://beast.community/workshop_continuous_diffusion_yfv 
chronumental --tree YFV.rooted.nwk  --dates YFV.metadata.tsv --tree_out YFV.timed.default_branch_random.nwk --steps 2000 --log_every 100 --lr 0.01  --treat_mutation_units_as_normalised_to_genome_size 10236 --clipped_adam  --locations YFV.metadata.tsv --variance_location 0.00016 --RRW



