
MODEL_CONFIGS = {
    "TFVideoMAE_S_16x224_FT": {
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_path_rate": 0.1,
    },

    "TFVideoMAE_B_16x224_FT": {
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_path_rate": 0.1,
    },

    "TFVideoMAE_B_16x384_FT": {
        "img_size": 384,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_path_rate": 0.1,
    },

    "TFVideoMAE_L_16x224_FT": {
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_path_rate": 0.1,
    },

    "TFVideoMAE_L_16x384_FT": {
        "img_size": 384,
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_path_rate": 0.1,
    },

    "TFVideoMAE_L_16x512_FT": {
        "img_size": 512,
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_path_rate": 0.1,
    },

    "TFVideoMAE_H_16x224_FT": {
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "drop_path_rate": 0.1,
    },
}