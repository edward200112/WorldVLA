关于 reserved-token 复用
这次没有改成“复用 Emu3 预留 id”。
结论是：当前保持“added special tokens + resize embeddings”方案更稳妥。没有足够证据证明 Emu3 存在一段可安全复用、且不会污染语义的 reserved ids，所以不做冒险覆盖。

python scripts/train_minimal.py --dataset_size 2 --num_epochs 1
python scripts/run_demo.py --checkpoint_dir outputs/unitok_drive_lite/checkpoint_last



4bit量化测试
python scripts/train_minimal.py --dataset_size 2 --num_epochs 1 --load_in_4bit
python scripts/run_demo.py --checkpoint_dir outputs/unitok_drive_lite/checkpoint_last --load_in_4bit
