from pyjapc import PyJapc


class AppContext:
    @property
    def japc(self) -> PyJapc:
        return PyJapc(
            incaAcceleratorName="SPS",
            selector="SPS.USER.ALL",
            noSet=True,
            logLevel="INFO",
        )


context = AppContext()
