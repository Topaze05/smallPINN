import json

from PDE_Models import SWE_2D
from models import AdaptedTRMPINN, MaskPINN, PINN_MLP, FF_PINN, TRM_PINN
from dataloader_ import SWEDataloader, generate_comparison_gif

def demo_adapted_TRM(dataset_path: str, n :int, T: int):
    dataloader = SWEDataloader(dataset_path, batch_size=50)
    model = AdaptedTRMPINN(3, 3, 64, 3, sigma=0.5, activation="tanh", num_latent_refinements=n,
                           num_refinement_blocks=T)
    swe_model = SWE_2D(model, dataloader, regularization_coef=1e-5, logs=True)
    acc1, acc2 = swe_model.train(dataloader, adam_epochs=200, lbfgs_epochs=10)

    results = swe_model.get_logs()
    results["acc1"] = acc1
    results["acc2"] = acc2

    with open("results_demo.json", "w") as f:
        json.dump(results, f)


    pred = swe_model.simulate()

    generate_comparison_gif(dataloader, pred, f"comparison_3d_n{n}_T{T}.gif", "3d")
    generate_comparison_gif(dataloader, pred, f"comparison_2d_n{n}_T{T}.gif", "2d")


if __name__ == "__main__":
    demo_adapted_TRM("path_of_your_dataset", 3, 5)
