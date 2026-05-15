import torch
import numpy as np

def kld_cpx_tosn(mu_real, mu_imag, logvar, tau_real, tau_imag, eps=1e-8):
    """
    KL divergence between complex distribution and standard CN(0,1).
    Ensures tau^2 <= sigma^4 for validity.
    """
    sigma_sq = torch.exp(logvar).clamp(min=eps)   
    sigma_sq_sq = sigma_sq**2                 
    
    tau_abs_sq = (tau_real**2 + tau_imag**2).clamp_min(eps)
    tau_abs_sq = torch.minimum(tau_abs_sq, sigma_sq_sq)

    curr_len = torch.sqrt(tau_abs_sq)           
    max_len = sigma_sq 

    # Rescale pseudo-covariance if it exceeds variance bounds
    ratio = torch.ones_like(curr_len)
    mask = curr_len > 0
    ratio[mask] = (max_len[mask] / curr_len[mask]).clamp(max=1.0)

    tau_real = tau_real * ratio
    tau_imag = tau_imag * ratio

    # Compute determinant and trace of the covariance matrix
    det_C = (sigma_sq_sq**2 - (tau_real**2 + tau_imag**2)).clamp(min=eps)
    trace_C = sigma_sq_sq
    mu_sq = mu_real**2 + mu_imag**2
    
    # Standard KL formula for complex normal
    kl = -0.5 * torch.log(det_C) + trace_C + mu_sq - 1.0
    return kl

def complex_kl_divergence(
    mu_p_real, mu_p_imag, var_p, pseudo_cov_p_real, pseudo_cov_p_imag,
    mu_q_real, mu_q_imag, var_q, pseudo_cov_q_real, pseudo_cov_q_imag
):
    # Construct 2x2 real covariance matrix from complex parameters
    def build_cov_matrix(var, C_r, C_i):
        return torch.array([
            [(var + C_r) / 2, C_i / 2],
            [C_i / 2, (var - C_r) / 2]
        ])

    Sigma_p = build_cov_matrix(var_p, pseudo_cov_p_real, pseudo_cov_p_imag)
    Sigma_q = build_cov_matrix(var_q, pseudo_cov_q_real, pseudo_cov_q_imag)

    if np.linalg.det(Sigma_p) <= 0 or np.linalg.det(Sigma_q) <= 0:
        raise ValueError("Covariance matrices must be positive definite.")

    Sigma_q_inv = np.linalg.inv(Sigma_q)
    delta_mu = np.array([mu_q_real - mu_p_real, mu_q_imag - mu_p_imag])

    # Standard multivariate normal KL components
    term_trace = np.trace(Sigma_q_inv @ Sigma_p)
    term_quadratic = delta_mu.T @ Sigma_q_inv @ delta_mu
    term_log = np.log(np.linalg.det(Sigma_q) / np.linalg.det(Sigma_p))

    return 0.5 * (term_trace + term_quadratic - 2 + term_log)

def kl_divergence_complexn(P_mu_real, P_mu_imag, P_logvar, P_tau_real, P_tau_imag,
                          Q_mu_real, Q_mu_imag, Q_logvar, Q_tau_real, Q_tau_imag,
                          eps=1e-4, debug=False, lambda_reg=0.0):
    """
    Computes symmetric KL divergence: KL(P||Q) + KL(Q||P)
    """
    def compute_kl(mu_real_p, mu_imag_p, logvar_p, tau_real_p, tau_imag_p,
                   mu_real_q, mu_imag_q, logvar_q, tau_real_q, tau_imag_q):
        
        sigma_p_sq = torch.exp(logvar_p.clamp(max=10)).clamp(min=eps)
        sigma_q_sq = torch.exp(logvar_q.clamp(max=10)).clamp(min=eps)

        # Constrain pseudo-covariance magnitude
        tau_abs_sq_p = torch.minimum(tau_real_p**2 + tau_imag_p**2, sigma_p_sq**2)
        scaling_p = torch.sqrt((tau_abs_sq_p / (tau_real_p**2 + tau_imag_p**2 + eps)) + eps)
        tau_real_p, tau_imag_p = tau_real_p * scaling_p, tau_imag_p * scaling_p

        tau_abs_sq_q = torch.minimum(tau_real_q**2 + tau_imag_q**2, sigma_q_sq**2)
        scaling_q = torch.sqrt((tau_abs_sq_q / (tau_real_q**2 + tau_imag_q**2 + eps)) + eps)
        tau_real_q, tau_imag_q = tau_real_q * scaling_q, tau_imag_q * scaling_q

        # Log-determinant and trace terms
        det_P = (sigma_p_sq**2 - (tau_real_p**2 + tau_imag_p**2)).clamp(min=eps)
        det_Q = (sigma_q_sq**2 - (tau_real_q**2 + tau_imag_q**2)).clamp(min=eps)
        log_det_ratio = torch.log(det_Q) - torch.log(det_P)
        trace_term = 2 * (sigma_q_sq * sigma_p_sq - (tau_real_q * tau_real_p + tau_imag_q * tau_imag_p)) / det_Q

        # Quadratic mean difference term
        d_r, d_i = mu_real_p - mu_real_q, mu_imag_p - mu_imag_q
        re_tauqd2 = tau_real_q * (d_r**2 - d_i**2) + 2 * tau_imag_q * d_r * d_i
        quad_term = 2 * (sigma_q_sq * (d_r**2 + d_i**2) - re_tauqd2) / det_Q

        return torch.clamp(0.5 * (log_det_ratio + trace_term + quad_term - 2.0), min=0.0)

    kl_pq = compute_kl(P_mu_real, P_mu_imag, P_logvar, P_tau_real, P_tau_imag,
                       Q_mu_real, Q_mu_imag, Q_logvar, Q_tau_real, Q_tau_imag)
    kl_qp = compute_kl(Q_mu_real, Q_mu_imag, Q_logvar, Q_tau_real, Q_tau_imag,
                       P_mu_real, P_mu_imag, P_logvar, P_tau_real, P_tau_imag)
    
    return kl_pq + kl_qp

def kl_divergence_normal_to_standard(mu, logvar):
    """
    KL divergence between N(mu, sigma^2) and standard N(0, 1).
    """
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
    return kl.mean()