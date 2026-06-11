import matplotlib.pyplot as plt
from matplotlib import rc
from functools import partial
import pandas as pd
import numpy as np

from src.pipeline.initialise import *
from src.pipeline.pipeline_steps import train_step, val_step
from src.metrics.unbiased_metrics import profile_generation
from src.utils.helper_functions import make_grid

# Set plot styling
rc("font", **{"family": "serif", "serif": ["Computer Modern"]}, size=12)
rc("text", usetex=True)

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

save_every = int(parser["PIPELINE"]["SAVE_EVERY"])
num_epochs = int(parser["PIPELINE"]["NUM_EPOCHS"])
min_samples = int(parser["GENERATION_EVAL"]["MIN_SAMPLES"])
max_samples = int(parser["GENERATION_EVAL"]["MAX_SAMPLES"])
num_points = int(parser["GENERATION_EVAL"]["NUM_POINTS"])
num_plot = int(parser["GENERATION_EVAL"]["NUM_PLOT"])


def run_experiment(exp_num, train_x, val_x, log_path):

    # Initialise the pipeline
    key = jax.random.PRNGKey(exp_num)
    key, EBM_params, EBM_fwd = init_EBM(key)
    key, GEN_params, GEN_fwd = init_GEN(key)
    EBM_optimiser, EBM_opt_state = init_EBM_optimiser(EBM_params)
    GEN_optimiser, GEN_opt_state = init_GEN_optimiser(GEN_params)

    # Tuple up for cleanliness
    params_tup = (EBM_params, GEN_params)
    fwd_fcn_tup = (EBM_fwd, GEN_fwd)
    optimiser_tup = (EBM_optimiser, GEN_optimiser)
    opt_state_tup = (EBM_opt_state, GEN_opt_state)
    del EBM_params, GEN_params, EBM_fwd, GEN_fwd, EBM_optimiser, GEN_optimiser, EBM_opt_state, GEN_opt_state

    # Preload the pipeline functions with immutable arguments
    loaded_train_step = partial(
        train_step,
        optimiser_tup=optimiser_tup,
        fwd_fcn_tup=fwd_fcn_tup,
    )
    loaded_val_step = partial(
        val_step, fwd_fcn_tup=fwd_fcn_tup,
    )

    metrics_fcn = partial(
        profile_generation,
        train_x=train_x[:max_samples].reshape(
            -1, train_x.shape[-3], train_x.shape[-2], train_x.shape[-1]
        ),  # Flatten batches
        val_x=val_x.reshape(
            -1, val_x.shape[-3], val_x.shape[-2], val_x.shape[-1]
        ),  # Flatten batches
        fwd_fcn_tup=fwd_fcn_tup,
        min_samples=min_samples,
        max_samples=max_samples,
        num_points=num_points,
        num_plot=num_plot,
    )

    # Jit the pipeline functions
    jit_train_step = jax.jit(loaded_train_step)
    jit_val_step = jax.jit(loaded_val_step)
    jit_metrics_fcn = jax.jit(metrics_fcn)

    # ### TODO: Add FLOPS calculation ##
    # if exp_num == 0:
    #     x = jnp.zeros((train_x.shape[-3], train_x.shape[-2], train_x.shape[-1]))
    #     compiled = jit_val_step.lower(jax.random.PRNGKey(0), x, params_tup).compile()
    #     flops = compiled.cost_analysis()[0]['flops']
    #     print(f"Number of FLOPS: {flops:.4e}")

    @jax.jit
    def train_batches(carry, x):
        """Batch training fcn for scanning."""
        key, params_tup, opt_state_tup = carry
        key, params_tup, opt_state_tup, loss, grad, var = jit_train_step(
            key, x, params_tup, opt_state_tup
        )
        return (key, params_tup, opt_state_tup), (loss, grad, var)

    @jax.jit
    def val_batches(carry, x, params_tup):
        """Batch validation fcn for scanning."""
        key = carry
        key, loss = jit_val_step(key, x, params_tup)
        return key, loss

    img_evolution = np.zeros(
        (num_epochs // save_every, val_x.shape[-3], val_x.shape[-2], val_x.shape[-1])
    )

    df = pd.DataFrame(
        columns=[
            "Epoch",
            "Train Loss",
            "Train Grad",
            "Train Grad Var",
            "Val Loss",
            "FID_inf",
            "MIFID_inf",
            "KID_inf",
            "IS_inf",
        ]
    )
    df.to_csv(f"{log_path}/experiment{exp_num}.csv", index=False)
    del df

    for epoch in range(num_epochs):

        # Train
        (key, params_tup, opt_state_tup), (train_loss, train_grad, train_grad_var) = jax.lax.scan(
            f=train_batches, init=(key, params_tup, opt_state_tup), xs=train_x
        )

        train_loss = train_loss.mean()
        train_grad = train_grad.mean()
        train_grad_var = train_grad_var.mean()

        # Validate
        key, val_loss= jax.lax.scan(
            f=partial(val_batches, params_tup=params_tup), init=key, xs=val_x
        )

        val_loss = val_loss.mean()

        # Profile generative capacity using unbiased metrics
        key, fid_inf, mifid_inf, kid_inf, is_inf, four_real, four_fake = jit_metrics_fcn(key, params_tup)
        
        # Save to dataframe
        epoch_df = pd.DataFrame(
            {
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Train Grad": train_grad,
                "Train Grad Var": train_grad_var,
                "Val Loss": val_loss,
                "FID_inf": fid_inf,
                "MIFID_inf": mifid_inf,
                "KID_inf": kid_inf,
                "IS_inf": is_inf,
            },
            index=[0],
        )

        epoch_df.to_csv(f"{log_path}/experiment{exp_num}.csv", mode="a", header=False, index=False)

        if (epoch % save_every == 0 or epoch == num_epochs-1) and exp_num == 0:
            fake_grid = make_grid(four_fake, n_row=4)
            real_grid = make_grid(four_real, n_row=4)

            # If fake grid is outside [0, 1], print warning
            if (fake_grid > 1).any() or (fake_grid < 0).any():
                print("clip", fake_grid.min(), fake_grid.max())

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(real_grid)
            ax[0].set_title("Real Images")
            ax[0].axis("off")
            ax[1].imshow(fake_grid)
            ax[1].set_title("Generated Images")
            ax[1].axis("off")
            plt.suptitle(
                f"Epoch: {epoch} \n\n"
                + r"$\overline{FID}_\infty$: "
                + f"{fid_inf:.4g}, "
                + r"$\overline{MIFID}_\infty$: "
                + f"{mifid_inf:.4g}, "
                + r"$\overline{KID}_\infty$: "
                + f"{kid_inf:.4g}"
                # + r"$\overline{IS}_\infty$: "
                # + f"{is_inf:.4g}"
            )
            plt.tight_layout()
            plt.savefig(f"{log_path}/images/{epoch}.png", dpi=750)

            img_evolution[epoch // save_every] = four_fake[0]

    if exp_num == 0:
        evol_grid = make_grid(img_evolution, n_row=1)
        plt.figure()
        plt.imshow(evol_grid)
        plt.axis("off")
        plt.title("Evolution of Generated Images")
        plt.tight_layout()
        plt.savefig(f"{log_path}/images/evolution.png", dpi=750)

    # Clean up
    del params_tup
    del fwd_fcn_tup
    del optimiser_tup
    del opt_state_tup
    del loaded_train_step
    del loaded_val_step
    del jit_train_step
    del jit_val_step
    del metrics_fcn
