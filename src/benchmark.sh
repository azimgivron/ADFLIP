


export PYTHONPATH=$PWD:$PYTHONPATH



# CUDA_VISIBLE_DEVICES=0 python3 test/validate_dataset.py \
#     --ckpt_path _ADFLIP_PDB_icml_hd128_bz12_noise0_dp0.1_label_smoothing_assembly_132700.pt \
#     --output validation_errors.txt

# CUDA_VISIBLE_DEVICES=0 python test/validate_failed.py \
#   --ckpt_path _ADFLIP_PDB_icml_hd128_bz12_noise0_dp0.1_label_smoothing_assembly_132700.pt \
#   --error_file errors.txt \
#   --sample_scheme adaptive --num_step 8 --temp 0.1 --noise 0.1




# CUDA_VISIBLE_DEVICES=2 python3 test/benchmark.py --sample_scheme fixed --pdb_type metal --ckpt_path ADFLIP_ICML_camera_ready_v1.pt
CUDA_VISIBLE_DEVICES=0 python3 test/benchmark.py --pdb_type metal --ckpt_path ADFLIP_ICML_camera_ready_v1.pt --thershold 0.9 --n_samples 1
# CUDA_VISIBLE_DEVICES=0 python3 test/benchmark.py --pdb_type nucleotide --ckpt_path ADFLIP_ICML_camera_ready_v1.pt --thershold 0.9 --n_samples 1
# CUDA_VISIBLE_DEVICES=0 python3 test/benchmark.py --pdb_type molecule --ckpt_path ADFLIP_ICML_camera_ready_v1.pt --thershold 0.9 --n_samples 1



# CUDA_VISIBLE_DEVICES=2 python3 test/benchmark.py --pdb_type nucleotide --ckpt_path ADFLIP_ICML_camera_ready_v1.pt


# CUDA_VISIBLE_DEVICES=2 python3 test/benchmark.py --pdb_type molecule --ckpt_path ADFLIP_ICML_camera_ready_v1.pt

# python3 test/benchmark.py --pdb_type nucleotide
# python3 test/benchmark.py --pdb_type molecule
