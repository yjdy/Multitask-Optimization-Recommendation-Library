from ..weighted_methods import WeightMethod
from ..utils import get_shared_grads
import torch

class IMTLG(WeightMethod):
    """TOWARDS IMPARTIAL MULTI-TASK LEARNING: https://openreview.net/pdf?id=IMPnRXEWpvr"""

    def __init__(self, n_tasks,model,optimizer, device: torch.device):
        super(IMTLG,self).__init__(n_tasks,
                         model=model,
                         optimizer=optimizer,
                         device=device)

    def get_weighted_loss(
            self,
            losses,
            shared_parameters,
            **kwargs,
    ):
        norm_grads = {}

        G = get_shared_grads(losses, self.model, self.optimizer)

        for i, grad in enumerate(G):
            norm_term = torch.norm(grad)
            norm_grads[i] = grad / norm_term

        GTG = torch.mm(G, G.t())

        D = (
                G[
                    0,
                ]
                - G[
                  1:,
                  ]
        )

        U = torch.stack(tuple(v for v in norm_grads.values()))
        U = (
                U[
                    0,
                ]
                - U[
                  1:,
                  ]
        )
        first_element = torch.matmul(
            G[
                0,
            ],
            U.t(),
        )
        try:
            second_element = torch.inverse(torch.matmul(D, U.t()))
        except:
            # workaround for cases where matrix is singular
            second_element = torch.inverse(
                torch.eye(self.n_tasks - 1, device=self.device) * 1e-8
                + torch.matmul(D, U.t())
            )

        alpha_ = torch.matmul(first_element, second_element)
        alpha = torch.cat(
            (torch.tensor(1 - alpha_.sum(), device=self.device).unsqueeze(-1), alpha_)
        )

        loss = torch.sum(losses * alpha)
        extra_outputs = {}
        extra_outputs["weights"] = alpha
        extra_outputs["GTG"] = GTG.detach().cpu().numpy()
        return loss, extra_outputs