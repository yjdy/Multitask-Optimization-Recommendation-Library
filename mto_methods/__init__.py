from mto_methods.loss_balance import FAMO, STL, ScaleInvariantLinearScalarization, LinearScalarization, Uncertainty, RLW,DynamicWeightAverage
from mto_methods.gradient_balance import NashMTL, IMTLG, LOG_IMTLG, MGDA, LOG_MGDA, CAGrad, PCGrad,GradDrop
from mto_methods.parameter_balancing.pub import ParameterUpdateBalancing

METHODS = dict(
    stl=STL,
    base=LinearScalarization,
    uw=Uncertainty,
    scaleinvls=ScaleInvariantLinearScalarization,
    rlw=RLW,
    dwa=DynamicWeightAverage,
    pcgrad=PCGrad,
    mgda=MGDA,
    graddrop=GradDrop,
    log_mgda=LOG_MGDA,
    cagrad=CAGrad,
    # log_cagrad=LOG_CAGrad,
    imtl=IMTLG,
    log_imtl=LOG_IMTLG,
    nashmtl=NashMTL,
    famo=FAMO,
    pub=ParameterUpdateBalancing
)
