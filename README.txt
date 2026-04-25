README.txt

This project tackles 15-class scene recognition on a dataset of 1,500 images using a progressive CNN-based approach. Starting from a custom CNN baseline (74.44% val accuracy), we systematically improved performance by exploring ResNet architectures from scratch (78.67%), then applying transfer learning with pretrained ResNet152 (95.33%) and ConvNeXt Small (97.67%). Ablation studies were conducted across model depth, unfreezing strategy, learning rate, data augmentation, and label smoothing. A per-class analysis revealed complementary strengths between the two architectures — ConvNeXt excelling at texture-rich indoor scenes, ResNet152 stronger on structured outdoor scenes — motivating a soft-voting ensemble that achieves a final validation accuracy of 98.67% and macro-average F1 of 0.9867.

Requirements:
    pip install torch torchvision scikit-learn pillow numpy

Files:
    scene_recog_cnn.py  - Main script with train() and test() functions
    model.py            - Model definitions (ConvNeXt Small and ResNet152)
    main.py             - Sample code for running model on test images
    trained_cnn.pth     - Pretrained ConvNeXt Small weights
    trained_cnn_res.pth - Pretrained ResNet152 weights

Usage:
    # Training (trains both ConvNeXt Small and ResNet152 as we use an Ensemble)
    python3 scene_recog_cnn.py --phase train --train_data_dir ./data/train --model_dir .

    # Testing (runs soft-voting ensemble of both models)
    python3 scene_recog_cnn.py --phase test --test_data_dir ./data/test --model_dir .

    # Or using sample code
    python3 main.py

Note:
    - model.py is a module imported by scene_recog_cnn.py, do not run it directly.
    - Both trained_cnn.pth and trained_cnn_res.pth must be in the same folder as scene_recog_cnn.py.
    - Test function runs a soft-voting ensemble of ConvNeXt Small and ResNet152.
