memory-wrap-edu/
│
├── README.md
│
├── data/
│   └── memory_bank.pt              # saved after build_stratified_memory.py runs
│
├── checkpoints/
│   ├── baseline/
│   │   └── best/                   # saved tokenizer + model after step 1
│   └── memory_wrap/
│       └── best_memory_wrap.pt     # saved after step 3
│
├── src/
│   ├── memory_wrap.py              # the Memory Wrap module
│   ├── train_baseline.py           # step 1
│   ├── build_stratified_memory.py  # step 2
│   └── train_memory_wrap.py        # step 3
│
└── requirements.txt