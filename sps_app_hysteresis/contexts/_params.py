from __future__ import annotations

from ._base_context import (
    EddyCurrentModel,
    MeasurementEddyCurrentModel,
    ParameterNames,
    RemoteParameterNames,
)

UCAP_NODE = "UCAP-NODE-SPS-HYSTCOMP"

MBI_PARAMS = ParameterNames(
    TRIGGER="XTIM.SX.FCY2500-CT/Acquisition",
    I_PROG="rmi://virtual_sps/MBI/IREF",
    B_PROG="rmi://virtual_sps/SPSBEAM/B",
    ADD_PROG_TRIGGER=f"rda3://{UCAP_NODE}/UCAP.MACHINE_MODE/Acquisition",
    B_CORRECTION=f"rda3://{UCAP_NODE}/UCAP.SPSBEAM/BHYS_CORRECTION",
    CYCLE_START="XTIM.SX.SCY-CT/Acquisition",
    I_PROG_DYNECO=f"rda3://{UCAP_NODE}/UCAP.MBI/DYNECO_IREF",
    I_PROG_FULLECO=f"rda3://{UCAP_NODE}/UCAP.MBI/FULLECO_IREF",
    FULLECO_TRIGGER="XTIM.SX.FCY-MMODE-CT/Acquisition",
    I_MEAS="MBI/LOG.I.MEAS",
    B_MEAS=f"rda3://{UCAP_NODE}/SPS.BTRAIN.BMEAS.SP/Acquisition",
    BDOT_PROG=f"rda3://{UCAP_NODE}/UCAP.SPSBEAM/BDOT",
    BDOT_MEAS="SR.BMEAS-SP-BDOT-SD/CycleSamples#samples",
    BDOT_PLAYED=f"rda3://{UCAP_NODE}/UCAP.SPSBEAM/BDOT_PLAYED",
    RESET_REFERENCE_TRIGGER="rmi://virtual_sps/SPSBEAM/B",
    LSA_TRIM_PARAM="SPSBEAM/BHYS",
)


MBI_REMOTE_PARAMS = RemoteParameterNames(
    CYCLE_WARNING="rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/CycleDataFCY",
    CYCLE_CORRECTION=(
        "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/CycleDataCorrection"
    ),
    CYCLE_MEASURED=(
        "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/CycleDataMeasRef"
    ),
    METRICS="rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/Metrics",
    RESET_REFERENCE=(
        "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/ResetReference"
    ),
    SET_GAIN="rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/Gain",
)

MBI_EDDY_CURRENT_MODEL = EddyCurrentModel(
    NAME="SPS.MBI.EDDY_CURRENT.3EXP",
    VERSION="1.0",
)

MBI_MEASUREMENT_EDDY_CURRENT_MODEL = MeasurementEddyCurrentModel(
    NAME="SPS.MBI.BTRAIN.EDDY_CURRENT",
    VERSION="1.0",
)
