from __future__ import annotations

import logging
import typing

from accwidgets import lsa_selector
from op_app_context import context
from qtpy import QtCore, QtGui, QtWidgets

from ...history import PredictionHistory
from ._view import HistoryPlotWidget

log = logging.getLogger(__name__)


class HistoryWidget(QtWidgets.QWidget):
    def __init__(
        self,
        history: PredictionHistory,
        parent: QtWidgets.QWidget | None = None,
        *,
        max_tabs: int = 10,
        measured_available: bool = False,
    ) -> None:
        super().__init__(parent=parent)

        self.measured_available = measured_available
        self.max_tabs = max_tabs
        self.LsaSelector = self._setup_lsa_selector()

        self.actionRefreshLsaSelector = QtWidgets.QAction(self)
        self.actionRefreshLsaSelector.setText("&Refresh LSA Selector")
        self.actionRefreshLsaSelector.triggered.connect(self.LsaSelector.model.refetch)

        self.actionImport_Predictions = QtWidgets.QAction(self)
        self.actionImport_Predictions.setText("&Import Predictions")
        self.actionExport_Predictions = QtWidgets.QAction(self)
        self.actionExport_Predictions.setText("&Export Predictions")
        self.actionExit = QtWidgets.QAction(self)
        self.actionExit.setText("&Exit")
        self.actionExit.triggered.connect(self.close)

        self.menubar = QtWidgets.QMenuBar(self)
        file_menu = self.menubar.addMenu("&File")
        file_menu.addAction(self.actionRefreshLsaSelector)
        file_menu.addAction(self.actionImport_Predictions)
        file_menu.addAction(self.actionExport_Predictions)
        file_menu.addSeparator()
        file_menu.addAction(self.actionExit)

        view_menu = self.menubar.addMenu("&View")
        self.actionResetAxes = QtWidgets.QAction(self)
        self.actionResetAxes.setText("&Reset Axes")
        view_menu.addAction(self.actionResetAxes)

        # make the widget
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addWidget(self.LsaSelector)
        self.tabWidget = QtWidgets.QTabWidget(self)
        self.layout().addWidget(self.tabWidget)

        # add menubar
        self.layout().setMenuBar(self.menubar)

        self._history = history
        self._tabs: dict[str, HistoryPlotWidget] = {}

        # make tabs closable
        self.tabWidget.setTabsClosable(True)
        self.tabWidget.tabCloseRequested.connect(self.onTabCloseRequested)

        # set minimum size
        if measured_available:
            self.tabWidget.setMinimumSize(1200, 600)
        else:
            self.tabWidget.setMinimumSize(800, 600)

        # set floating
        self.setWindowFlags(QtCore.Qt.Window)

    def _setup_lsa_selector(self) -> lsa_selector.LsaSelector:
        selector_model = lsa_selector.LsaSelectorModel(
            accelerator=lsa_selector.LsaSelectorAccelerator.SPS,
            lsa=context.lsa_client,
            categories={
                lsa_selector.AbstractLsaSelectorContext.Category.MD,
                lsa_selector.AbstractLsaSelectorContext.Category.OPERATIONAL,
            },
        )
        LsaSelector = lsa_selector.LsaSelector(model=selector_model, parent=self)
        LsaSelector.contextSelectionChanged.connect(self.onUserChanged)
        LsaSelector.setMaximumWidth(300)
        LsaSelector.setMinimumWidth(300)

        return LsaSelector

    @QtCore.Slot(lsa_selector.AbstractLsaSelectorContext)
    def onUserChanged(self, user: lsa_selector.AbstractLsaSelectorContext) -> None:
        # if button state is off, trigger the button
        log.debug(f"Requested to show {user.name}")
        self.show_or_create_tab(user.name)

    @QtCore.Slot()
    def onResetAxes(self) -> None:
        if self.tabWidget.currentWidget() is not None:
            typing.cast(HistoryPlotWidget, self.tabWidget.currentWidget()).resetAxes()

    def show_or_create_tab(self, name: str) -> None:
        if name not in self._tabs:
            msg = f"Creating new tab for {name}"
            log.debug(msg)
            widget = HistoryPlotWidget(
                self._history.model(name),
                self,
                plot_measured=self.measured_available,
            )
            self.tabWidget.addTab(widget, name)
            self.tabWidget.setCurrentWidget(widget)
            self._tabs[name] = widget

            if len(self._tabs) > self.max_tabs:
                msg = f"Maximum number of tabs reached ({self.max_tabs}). Close some tabs to open new ones."
                log.warning(msg)
                return

        elif self.tabWidget.indexOf(self._tabs[name]) == -1:
            msg = f"Tab {name} was closed, reopening."
            log.debug(msg)

            self.tabWidget.addTab(self._tabs[name], name)
        else:
            msg = f"Tab {name} already open, switching to it."
            log.debug(msg)

            self.tabWidget.setCurrentWidget(self._tabs[name])

    def onTabCloseRequested(self, index: int) -> None:
        self.tabWidget.widget(index)
        name = self.tabWidget.tabText(index)

        log.debug(f"Closing tab {name}")

        # keep tab internally, but close the tab
        self.tabWidget.removeTab(index)

        if len(self._tabs) > self.max_tabs:
            self._tabs.pop(name)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # close window only if closed programmatically, otherwise hide
        if event.spontaneous():
            event.accept()
        else:
            self.hide()
            event.ignore()
