from loss_balance import FAMO, STL, ScaleInvariantLinearScalarization, LinearScalarization, Uncertainty, RLW
from gradient_balance import NashMTL, IMTLG, LOG_IMTLG, MGDA, LOG_MGDA, CAGrad, PCGrad
from pub import ParameterUpdateBalancing

METHODS = dict(
    stl=STL,
    ls=LinearScalarization,
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
