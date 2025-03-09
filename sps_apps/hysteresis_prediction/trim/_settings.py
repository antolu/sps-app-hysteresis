from __future__ import annotations

import typing

import pyda
from op_app_context import context
from qtpy import QtCore

from .._metadata import APP_NAME, ORGANIZATION_NAME


class TrimSettings(QtCore.QObject):
    trimEnabledChanged = QtCore.Signal(str, bool)
    """ Emitted when the trim is enabled or disabled for a cycle """

    trimDryRunChanged = QtCore.Signal(str, bool)
    """ Emitted when the trim is a dry run or not for a cycle """

    trimStartChanged = QtCore.Signal(str, float)
    """ Emitted when the trim start time is changed for a cycle """

    trimEndChanged = QtCore.Signal(str, float)
    """ Emitted when the trim end time is changed for a cycle """

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)

    def trim_enabled(self) -> typing.Mapping[str, bool]:
        """Whether the trim is enabled or not."""
        raise NotImplementedError

    def dry_run(self) -> typing.Mapping[str, bool]:
        """Whether the trim is a dry run or not."""
        raise NotImplementedError

    def trim_start(self) -> typing.Mapping[str, float]:
        """CTime of the start of the trim. Normally should not exceed beam injection time."""
        raise NotImplementedError

    def trim_end(self) -> typing.Mapping[str, float]:
        """CTime of the end of the trim. Normally should not exceed beam injection time."""
        raise NotImplementedError


class LocalTrimSettingsContainer(typing.MutableMapping[str, bool | float]):
    def __init__(self, settings: QtCore.QSettings, *, key: str):
        self.settings = settings
        self.key = key

    def __getitem__(self, cycle_name: str) -> bool | float:
        if self.key.endswith("initial_trim_enabled"):
            """ Default value is always False """
            if not self.settings.contains(self.key):
                self.settings.setValue(self.key, {})
            elif not self.settings.value(self.key).get(cycle_name, None):
                settings = self.settings.value(self.key)
                settings[cycle_name] = False
                self.settings.setValue(self.key, settings)

            return False

        value = self.settings.value(self.key, {}).get(cycle_name, None)
        if value is None:
            raise KeyError(cycle_name)

        return value

    def __setitem__(self, cycle_name: str, value: bool | float) -> None:
        if self.key.endswith("initial_trim_enabled"):
            msg = "Cannot set initial trim enabled value locally"
            raise NotImplementedError(msg)

        settings = self.settings.value(self.key, {})
        settings[cycle_name] = value
        self.settings.setValue(self.key, settings)

    def __delitem__(self, cycle_name: str) -> None:
        settings = self.settings.value(self.key, {})
        del settings[cycle_name]
        self.settings.setValue(self.key, settings)

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.settings.value(self.key, {}))

    def __len__(self) -> int:
        return len(self.settings.value(self.key, {}))


class LocalTrimSettings(TrimSettings):
    def __init__(self, parent: QtCore.QObject | None = None, *, prefix: str):
        super().__init__(parent)

        self.prefix = prefix

        self.settings = QtCore.QSettings(
            ORGANIZATION_NAME,
            APP_NAME,
        )
        self.settings.sync()

    def trim_enabled(self) -> typing.MutableMapping[str, bool]:
        return LocalTrimSettingsContainer(  # type: ignore[return-value]
            self.settings, key=f"{self.prefix}/trim_enabled"
        )

    def initial_trim_enabled(self) -> typing.MutableMapping[str, bool]:
        return LocalTrimSettingsContainer(  # type: ignore[return-value]
            self.settings, key=f"{self.prefix}/initial_trim_enabled"
        )

    def dry_run(self) -> typing.MutableMapping[str, bool]:
        return LocalTrimSettingsContainer(self.settings, key=f"{self.prefix}/dry_run")  # type: ignore[return-value]

    def trim_start(self) -> typing.MutableMapping[str, float]:
        return LocalTrimSettingsContainer(
            self.settings, key=f"{self.prefix}/trim_start"
        )

    def trim_end(self) -> typing.MutableMapping[str, float]:
        return LocalTrimSettingsContainer(self.settings, key=f"{self.prefix}/trim_end")


class OnlineTrimSettingsContainer(typing.MutableMapping[str, bool | float]):
    def __init__(
        self,
        da: pyda.SimpleClient,
        *,
        device_name: str,
        property_name: str,
        field_name: str,
    ):
        self._da = da

    def __getitem__(self, cycle_name: str) -> bool | float:
        raise NotImplementedError

    def __setitem__(self, cycle_name: str, value: bool | float) -> None:
        raise NotImplementedError

    def __delitem__(self, cycle_name: str) -> None:
        raise NotImplementedError

    def __iter__(self) -> typing.Iterator[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class OnlineTrimSettings(TrimSettings):
    def __init__(
        self,
        parent: QtCore.QObject | None = None,
        *,
        device: str,
    ):
        super().__init__(parent)

        self._da = pyda.SimpleClient(provider=context.japc_provider)
        self.device = device

    def trim_enabled(self) -> typing.MutableMapping[str, bool]:
        return OnlineTrimSettingsContainer(  # type: ignore[return-value]
            self._da,
            device_name="SPS.TRIM",
            property_name="Enabled",
            field_name="enabled",
        )

    def initial_trim_enabled(self) -> typing.MutableMapping[str, bool]:
        return OnlineTrimSettingsContainer(  # type: ignore[return-value]
            self._da,
            device_name="SPS.TRIM",
            property_name="initial_enabled",
            field_name="cycle",
        )

    def dry_run(self) -> typing.MutableMapping[str, bool]:
        return OnlineTrimSettingsContainer(  # type: ignore[return-value]
            self._da,
            device_name="SPS.TRIM",
            property_name="dry_run",
            field_name="cycle",
        )

    def trim_start(self) -> typing.MutableMapping[str, float]:
        return OnlineTrimSettingsContainer(
            self._da,
            device_name="SPS.TRIM",
            property_name="start_time",
            field_name="cycle",
        )

    def trim_end(self) -> typing.MutableMapping[str, float]:
        return OnlineTrimSettingsContainer(
            self._da,
            device_name="SPS.TRIM",
            property_name="end_time",
            field_name="cycle",
        )
