VARIANTS = ["wells_seismic", "wells_only", "seismic_only", "unconditional"]

VARIANT_LABELS = {
    "wells_seismic": "Wells + Seismic",
    "wells_only": "Wells Only",
    "seismic_only": "Seismic Only",
    "unconditional": "Unconditional",
}

EMBEDDING_METHODS = ["mds", "umap", "isomap", "tsne"]

EMBEDDING_LABELS = {
    "mds": "MDS",
    "umap": "UMAP",
    "isomap": "Isomap",
    "tsne": "t-SNE",
}

# Categorized hyperparameters to show in the report
HPARAM_CATEGORIES = {
    "🎯 Dataset & Setup": [
        ("num_iter", "Training epochs"),
        ("num_train_pyramids", "Training pyramids"),
        ("batch_size", "Batch size"),
        ("manual_seed", "Manual seed"),
        ("use_wells", "Uses wells"),
        ("use_seismic", "Uses seismic"),
        ("use_rock_physics", "Uses Rock Physics"),
    ],
    "⚙️ Pyramid Architecture": [
        ("stop_scale", "Stop scale"),
        ("num_parallel_scales", "Parallel scales"),
        ("scale0_noise_amp", "Scale-0 noise amp"),
        ("min_noise_amp", "Min noise amp"),
    ],
    "⚖️ Loss & Consistency Penalties": [
        ("rec_facies_loss_penalty", "Reconstruction weight (α)"),
        ("rec_rock_physics_loss_penalty", "Rock Physics Rec weight"),
        ("well_loss_penalty", "Well loss weight"),
        ("seismic_loss_penalty", "Seismic consistency weight"),
        ("elastic_loss_penalty", "Elastic consistency weight"),
        ("tv_loss_penalty", "TV loss weight"),
        ("diversity_loss_penalty", "Diversity loss weight"),
    ],
    "⚡ Optimization & WGAN-GP / SN": [
        ("lr_g", "Generator LR"),
        ("lr_d", "Discriminator LR"),
        ("discriminator_steps", "Discriminator steps"),
        ("generator_steps", "Generator steps"),
        ("gradient_loss_penalty", "WGAN GP weight"),
        ("use_gradnorm", "GradNorm balance"),
    ],
    "🌐 Physics Properties": [
        ("wavelet_f_peak", "Wavelet peak freq (Hz)"),
    ],
}
