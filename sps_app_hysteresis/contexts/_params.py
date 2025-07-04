from __future__ import annotations

from ._base_context import EddyCurrentModel, ParameterNames, RemoteParameterNames

MBI_PARAMS = ParameterNames(
    TRIGGER="SX.CZERO-CTML/CycleWarning",
    I_PROG="rmi://virtual_sps/MBI/IREF",
    B_PROG="rmi://virtual_sps/SPSBEAM/B",
    ADD_PROG_TRIGGER="rda3://UCAP-NODE-SPS-HYSTCOMP/UCAP.SX.INJ-100/Acquisition",
    B_CORRECTION="rda3://UCAP-NODE-SPS-HYSTCOMP/SPSBEAM.BHYS-CORRECTION/Acquisition",
    CYCLE_START="XTIM.SX.SCY-CT/Acquisition",
    I_PROG_DYNECO="rda3://UCAP-NODE-SPS-HYSTCOMP/SPS.MBI.DYNECO/IREF",
    I_PROG_FULLECO="rda3://UCAP-NODE-SPS-HYSTCOMP/SPS.MBI.FULLECO/IREF",
    FULLECO_TRIGGER="XTIM.SX.FCY-MMODE-CT/Acquisition",
    I_MEAS="MBI/LOG.I.MEAS",
    B_MEAS="SR.BMEAS-SP-B-SD/CycleSamples#samples",
    RESET_REFERENCE_TRIGGER="rmi://virtual_sps/SPSBEAM/B",
)


MBI_REMOTE_PARAMS = RemoteParameterNames(
    CYCLE_WARNING="rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/CycleDataFCY",
    CYCLE_CORRECTION=(
        "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/CycleDataCorrection"
    ),
    CYCLE_MEASURED=(
        "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/CycleDataMeasRef"
    ),
    RESET_REFERENCE=(
        "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/ResetReference"
    ),
    SET_GAIN="rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/Gain",
)

MBI_EDDY_CURRENT_MODEL = EddyCurrentModel(
    NAME="SPS.MBI.EDDY_CURRENT",
    VERSION="0.1",
)
