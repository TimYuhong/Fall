"""Training-time utilities for manifest building and model packaging.

Modules:
    generate_label_template  - Scan raw .bin files, emit JSONL label template
    build_manifest           - Scan extracted feature dirs, emit training JSONL
    export_model             - Package a trained model with model_meta.json

Typical workflow::

    # 1. Generate label template from raw .bin files
    python -m training.generate_label_template --data-root F:/Data_bin --out labels.jsonl

    # 2. (Manually review and correct labels.jsonl)

    # 3. Extract RD/RA/RE/PC features
    python -m offline.feature_extractor --manifest labels.jsonl --out F:/Features

    # 4. Build training manifest
    python -m training.build_manifest --features-root F:/Features --out train.jsonl

    # 5. (Train model ...)

    # 6. Package model for deployment
    python -m training.export_model --model model.pt --out F:/deploy --cfg config/Radar.cfg
"""
