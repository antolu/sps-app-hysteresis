from __future__ import annotations

from ._base_context import ParameterNames

MBI_PARAMS = ParameterNames(
    TRIGGER="SX.CZERO-CTML/CycleWarning",
    I_PROG="rmi://virtual_sps/MBI/IREF",
    B_PROG="rmi://virtual_sps/SPSBEAM/B",
    ADD_PROG_TRIGGER="rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/XTIM.UCAP.SCY-CT-500/Acquisition",
    B_CORRECTION="rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPSBEAM.BHYS-CORRECTION/Acquisition",
    CYCLE_START="XTIM.SX.SCY-CT/Acquisition",
    I_PROG_DYNECO="rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.MBI.DYNECO/IREF",
    I_PROG_FULLECO="rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.MBI.FULLECO/IREF",
    FULLECO_TRIGGER="XTIM.SX.FCY-MMODE-CT/Acquisition",
    I_MEAS="MBI/LOG.I.MEAS",
    B_MEAS="SR.BMEAS-SP-B-SD/CycleSamples#samples",
)
