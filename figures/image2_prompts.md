# image2 / gpt-image-2 prompt pack for PA-DSDB figures

These prompts are optional. The manuscript already contains compile-ready TikZ vector figures. Use these prompts if you want to generate polished bitmap or PDF-like concept art with an image2/gpt-image-2 workflow, then replace the TikZ panels manually after reviewing scientific accuracy.

## Fig. 1 Pipeline overview

Create a clean IEEE Transactions on Image Processing style scientific diagram for a method named "Physics-Anchored Degradation-State Diffusion Bridge (PA-DSDB) for Metalens Photography". Use a white background, subtle blue/teal/orange accents, crisp vector-like shapes, and professional typography. Show this left-to-right pipeline: metalens measurement y -> calibrated metalens operator H_kappa and H_kappa^T -> clean-domain anchor mu=R0(y,H_kappa) and optical uncertainty prior rho=P_psi(y,u,d_kappa) -> bridge state x_t -> dynamic degradation state s_t -> local reverse carrier Pi_t=epsilon_0+rho*s_t -> carrier network F_theta -> state-modulated reverse bridge update -> measurement-consistency projection -> restored image x_hat_0. Visually separate "fixed forward uncertainty" and "dynamic reverse control" with translucent grouped boxes. Do not include fake numerical results. Make it suitable for a top-tier IEEE journal paper.

## Fig. 2 Concept panels

Create a three-panel scientific figure with matching style. Panel (a): field-dependent metalens PSF spread increasing from center to edge, with small elliptical PSFs and a smooth curve. Panel (b): dynamic state dissipation over reverse sampling, showing reliable center regions decaying faster and off-axis regions staying uncertain longer. Panel (c): measurement projection loop x_hat -> H_kappa -> residual H_kappa(x_hat)-y -> H_kappa^T -> correction. Use clean labels and a restrained palette.

## Fig. 3 Bridge geometry

Create a polished comparison figure with three horizontal rows: generic diffusion (Gaussian noise to clean image, global stochasticity), raw-observation bridge (raw metalens y to clean x0 with a red domain mismatch warning), and PA-DSDB (raw y mapped by R0(y,H_kappa) to clean anchor mu, then a heteroscedastic clean-domain bridge with fixed rho to x0, dynamic s_t only in reverse). Emphasize that PA-DSDB avoids domain mismatch and separates forward uncertainty from reverse control.

## Fig. 4 Network architecture

Create an IEEE-style architecture diagram for one PA-DSDB reverse step. Modules: R0 anchor generator, P_psi optical prior estimator, G_eta state estimator, F_theta carrier network, measurement consistency projector. Inputs: y, u, d_kappa, x_t, residual cues r_t. Outputs: mu, rho, s_t, Pi_t, c_hat_t, x_{t-1}. Use clear arrows, grouped modules, and no decorative clutter.

## Fig. 5 Experiment evidence panels

Create a clean three-panel figure layout (not fake data): region masks for center/edge/off-axis evaluation, calibration drift robustness curve placeholders, and sampling step quality-cost tradeoff placeholders. Use neutral axes and labels without fabricated numeric values. The goal is a publication-ready placeholder layout that will later be replaced by measured curves.
